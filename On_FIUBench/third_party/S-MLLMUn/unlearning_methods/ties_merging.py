import argparse
import torch
import copy
from peft import PeftModel
from transformers import LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from peft.tuners.lora import Linear as LoraLinear
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']

def load_model_and_adapters(args):
    if "qwen2.5" in args.model_id.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.original_dir,
                                                                   torch_dtype=torch.float16,
                                                                   device_map="auto",
                                                                   attn_implementation="sdpa")
    elif "llava-onevision" in args.model_id.lower():
        print("Loading LLAVA-OneVision model...")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.original_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        raise ValueError("Model ID not recognized or not supported.")
    
    print(f"Loading Adapter 1: {args.MFA_multi}")
    model = PeftModel.from_pretrained(model, args.MFA_multi, adapter_name="adapter_1")
    print(f"Loading Adapter 2: {args.MFA_text}")
    model.load_adapter(args.MFA_text, adapter_name="adapter_2")
    print(f"Loading Adapter 3: {args.RA}")
    model.load_adapter(args.RA, adapter_name="adapter_3")
    return model

def get_delta_w_map(model, adapter_name):
    prefix = "base_model.model."
    model.set_adapter(adapter_name)
    delta_w_map, lora_layer_names = {}, set()
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            clean_name = name.replace(prefix, "")
            # Check if this adapter is active/present in this module
            if adapter_name not in module.lora_A:
                continue
                
            lora_A = module.lora_A[adapter_name].weight.data
            lora_B = module.lora_B[adapter_name].weight.data
            scaling = module.scaling[adapter_name]
            
            delta_w = scaling * (lora_B @ lora_A)
            delta_w_map[clean_name] = delta_w.cpu()
            lora_layer_names.add(clean_name)
            
    torch.cuda.empty_cache()
    return delta_w_map, lora_layer_names

def ties_merging(delta_ws_list, density):
    """
    Apply TIES-Merging on a list of delta_w tensors.
    """
    if not delta_ws_list:
        return None
        
    # Stack tensors: (num_models, ...)
    stacked = torch.stack(delta_ws_list) 
    
    # 1. TRIM: Keep top-k parameters by magnitude
    if density < 1.0:
        masks = []
        abs_stacked = torch.abs(stacked)
        for i in range(stacked.shape[0]):
            vec = abs_stacked[i]
            flat = vec.view(-1)
            if flat.numel() > 0:
                # Calculate threshold using sort to avoid "tensor too large" error with quantile
                k = int((1 - density) * flat.numel())
                k = max(0, min(k, flat.numel() - 1))
                threshold = torch.sort(flat)[0][k]
                
                mask = vec >= threshold
            else:
                mask = torch.ones_like(vec, dtype=torch.bool)
            masks.append(mask)
        mask_stacked = torch.stack(masks)
        stacked_trimmed = stacked * mask_stacked
    else:
        stacked_trimmed = stacked

    # 2. ELECT SIGN: Vote on the sign
    # Sum the signs of the trimmed vectors
    sign_tally = torch.sign(stacked_trimmed).sum(dim=0)
    elected_sign = torch.sign(sign_tally)
    
    # 3. DISJOINT MERGE: Average values that agree with the elected sign
    # Mask where sign matches the elected sign (and elected sign is not 0)
    mask_match = (torch.sign(stacked_trimmed) == elected_sign) & (elected_sign != 0)
    
    values_matching = stacked_trimmed * mask_match
    
    # Count contributors for averaging
    counts = mask_match.sum(dim=0)
    # Avoid division by zero
    counts = torch.max(counts, torch.ones_like(counts))
    
    merged_delta = values_matching.sum(dim=0) / counts
    return merged_delta

def main(args):
    model = load_model_and_adapters(args)
    
    print("Calculating Delta W for adapters...")
    delta_w_1, names_1 = get_delta_w_map(model, "adapter_1")
    delta_w_2, names_2 = get_delta_w_map(model, "adapter_2")
    delta_w_3, names_3 = get_delta_w_map(model, "adapter_3")
    
    all_names = names_1 | names_2 | names_3
    print(f"Total layers to merge: {len(all_names)}")
    
    final_delta_w_map = {}
    print(f"Applying TIES-Merging with density {args.density}...")
    
    for name in all_names:
        # Collect deltas, handling missing layers as zeros
        layer_deltas = []
        
        # We need a reference tensor for shape if a layer is missing in the first ones
        ref_tensor = None
        if name in delta_w_1: ref_tensor = delta_w_1[name]
        elif name in delta_w_2: ref_tensor = delta_w_2[name]
        elif name in delta_w_3: ref_tensor = delta_w_3[name]
        
        if ref_tensor is None: continue
        
        for d_map in [delta_w_1, delta_w_2, delta_w_3]:
            if name in d_map:
                layer_deltas.append(d_map[name])
            else:
                layer_deltas.append(torch.zeros_like(ref_tensor))
        
        merged_delta = ties_merging(layer_deltas, args.density)
        final_delta_w_map[name] = merged_delta

    # Unload adapters to get base model
    print("Unloading adapters and applying merged weights...")
    final_model = model.unload()
    final_state_dict = final_model.state_dict()
    
    applied_count = 0
    for name, module in final_model.named_modules():
        if name in final_delta_w_map:
            weight_key = f"{name}.weight"
            if weight_key in final_state_dict:
                base_weight = final_state_dict[weight_key]
                device = base_weight.device
                delta = final_delta_w_map[name].to(device)
                
                # Add merged delta to base weights
                # Note: TIES merging provides the task vector \tau.
                # The final weight is \theta_{pre} + \tau.
                final_state_dict[weight_key] = base_weight + delta
                applied_count += 1
                
    print(f"Merged weights applied to {applied_count} layers.")
    
    final_model.load_state_dict(final_state_dict)
    print(f"Model saved to: {args.save_dir}")
    final_model.save_pretrained(args.save_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(description="TIES-Merging")
    parser.add_argument("--model_id",
                        type=str,
                        default="model/base_model/llava-onevision-qwen2-7b-si-hf",
                        help="Pretrained model ID for config/loading checks")
    parser.add_argument("--original_dir",
                        type=str,
                        default="model/original_model/llava-onevision-qwen2-7b-original",
                        help="Base model directory")
    parser.add_argument("--save_dir",
                        type=str,
                        default="model/unlearned_model/llava-onevision/split_ratio_5/ties_merged0.2")
    # Adapter paths
    parser.add_argument("--MFA_text",
                        type=str,
                        required=True,
                        help="Path to first adapter")
    parser.add_argument("--MFA_multi",
                        type=str,
                        required=True,
                        help="Path to second adapter")
    parser.add_argument("--RA",
                        type=str,
                        required=True,
                        help="Path to third adapter")
    
    parser.add_argument("--density",
                        type=float,
                        default=0.5,
                        help="Density (k) for TIES merging (fraction of parameters to keep)")
                        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
