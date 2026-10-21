import argparse
import torch

from peft import PeftModel
from transformers import LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from peft.tuners.lora import Linear as LoraLinear



def load_model_and_adapter(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
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
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")
    model = PeftModel.from_pretrained(model, args.MFA_multi, adapter_name="MFA_multi")
    model.load_adapter(args.MFA_text, adapter_name="MFA_text")
    model.load_adapter(args.RA, adapter_name="RA")
    return model


def get_delta_w_map(model, adapter_name):
    prefix = "base_model.model."

    model.set_adapter(adapter_name)
    delta_w_map, lora_layer_names = {}, set()
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            clean_name = name.replace(prefix, "")
            active_adapter = module.active_adapter[0]
            if active_adapter not in module.lora_A:
                continue
            lora_A = module.lora_A[active_adapter].weight.data
            lora_B = module.lora_B[active_adapter].weight.data
            scaling = module.scaling[active_adapter]

            delta_w = scaling * (lora_B @ lora_A)

            delta_w_map[clean_name] = delta_w.cpu()
            lora_layer_names.add(clean_name)
    torch.cuda.empty_cache()
    return delta_w_map, lora_layer_names

def calc_mask_ratio(final_mask_map, delta_w_map):
    total_params = 0
    masked_params = 0
    for name, mask in final_mask_map.items():
        if name in delta_w_map:
            total_params += mask.numel()
            masked_params += mask.sum().item()
    ratio = masked_params / total_params if total_params > 0 else 0
    return masked_params, total_params, ratio


def main(args):
    model = load_model_and_adapter(args)
    print("Calculating Delta W for both LoRAs...")
    delta_w_RA_map, lora_layer_names = get_delta_w_map(model, "RA")
    delta_w_MFA_multi_map, _ = get_delta_w_map(model, "MFA_multi")
    delta_w_MFA_text_map, _ = get_delta_w_map(model, "MFA_text")

    print(f"Generating compound mask based on sign conflict AND relative magnitude with k={args.multi_k},{args.text_k}...")
    final_mask_multi_map = {}
    final_mask_text_map = {}
    for name, delta_w_RA in delta_w_RA_map.items():
        if name in delta_w_MFA_multi_map:
            delta_w_MFA_multi = delta_w_MFA_multi_map[name]
            norm_ratio_multi = torch.norm(delta_w_MFA_multi) / (torch.norm(delta_w_RA) + 1e-12)
            conflict_mask_multi = (delta_w_RA * delta_w_MFA_multi) < 0
            if args.multi_k < 0:
                final_mask_multi = conflict_mask_multi
            else:
                relative_mask_multi = args.multi_k * norm_ratio_multi * torch.abs(delta_w_RA) < torch.abs(delta_w_MFA_multi)
                final_mask_multi = conflict_mask_multi & relative_mask_multi

            final_mask_multi_map[name] = final_mask_multi

        if name in delta_w_MFA_text_map:
            delta_w_MFA_text = delta_w_MFA_text_map[name]
            norm_ratio_text = torch.norm(delta_w_MFA_text) / (torch.norm(delta_w_RA) + 1e-12)
            conflict_mask_text = (delta_w_RA * delta_w_MFA_text) < 0
            if args.text_k < 0:
                final_mask_text = conflict_mask_text
            else:
                relative_mask_text = args.text_k * norm_ratio_text * torch.abs(delta_w_RA) < torch.abs(delta_w_MFA_text)
                final_mask_text = conflict_mask_text & relative_mask_text

            final_mask_text_map[name] = final_mask_text

    masked_multi, total_multi, ratio_multi = calc_mask_ratio(final_mask_multi_map, delta_w_MFA_multi_map)
    masked_text, total_text, ratio_text = calc_mask_ratio(final_mask_text_map, delta_w_MFA_text_map)
    print(f"[multi] masked parameters: {masked_multi}/{total_multi}, ratio: {ratio_multi:.4f}")
    print(f"[text]  masked parameters: {masked_text}/{total_text}, ratio: {ratio_text:.4f}")

    delta_w_multi_sculpt_map = {}
    for name, delta_w_multi in delta_w_MFA_multi_map.items():
        if name in final_mask_multi_map:
            mask = final_mask_multi_map[name]
            pruned_delta_w = delta_w_multi.clone()
            pruned_delta_w[mask] = 0.0
            delta_w_multi_sculpt_map[name] = pruned_delta_w
        else:
            delta_w_multi_sculpt_map[name] = delta_w_multi

    delta_w_text_sculpt_map = {}
    for name, delta_w_text in delta_w_MFA_text_map.items():
        if name in final_mask_text_map:
            mask = final_mask_text_map[name]
            pruned_delta_w = delta_w_text.clone()
            pruned_delta_w[mask] = 0.0
            delta_w_text_sculpt_map[name] = pruned_delta_w
        else:
            delta_w_text_sculpt_map[name] = delta_w_text

    final_model = model.unload()
    final_state_dict = final_model.state_dict()

    print("Merging the final masked forget_lora into the base model...")
    for name, module in final_model.named_modules():
        if name in lora_layer_names:
            base_weight = final_state_dict[f"{name}.weight"].clone()
            device = base_weight.device
            delta_multi_sculpt = delta_w_multi_sculpt_map.get(name, torch.zeros_like(base_weight))
            delta_text_sculpt = delta_w_text_sculpt_map.get(name, torch.zeros_like(base_weight))

            delta_weight = (delta_multi_sculpt + delta_text_sculpt) / 2
            final_weight = base_weight + delta_weight.to(device)
            final_state_dict[f"{name}.weight"] = final_weight

    final_model.load_state_dict(final_state_dict)
    print(f"Model saved to: {args.save_dir}")
    final_model.save_pretrained(args.save_dir)




def parse_arguments():
    parser = argparse.ArgumentParser(description="sculpting")
    parser.add_argument("--model_id",
                        type=str,
                        default="model/base_model/llava-onevision-qwen2-7b-si-hf",
                        help="Pretrained model ID")
    parser.add_argument("--original_dir",
                        type=str,
                        default="model/original_model/llava-onevision-qwen2-7b-original",
                        help="Pretrained model ID")
    parser.add_argument("--save_dir",
                        type=str,
                        default="model/unlearned_model/llava-onevision/split_ratio_5/SMFA/sculpted")
    parser.add_argument("--MFA_text",
                        type=str,
                        default="model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/MFA_text")
    parser.add_argument("--MFA_multi",
                        type=str,
                        default="model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/MFA_multi")
    parser.add_argument("--RA",
                        type=str,
                        default="model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/RA")
    parser.add_argument("--multi_k", type=float, required=True)
    parser.add_argument("--text_k", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
