import os
import sys
import copy
import math
import time
import torch
import torch.nn as nn
import transformers
import argparse
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer,\
    LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from accelerate import Accelerator
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']

sys.path.append(('./'))
sys.path.append(('../'))
sys.path.append(('../../'))

from data_process.data_preprocess import MLLM_Dataset, train_collate_fn

DEBUG = False 
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SparseGPT:
    def __init__(self, layer, pretrain_layer):
        self.layer = layer
        self.pretrain_layer = pretrain_layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)           
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01, tops_mask=None, use_tops=False
    ):
        W = self.layer.weight.data.clone()           
        pretrain_weights = self.pretrain_layer.weight.data.clone()
        pretrain_weights = pretrain_weights.to(dtype=W.dtype, device=W.device)
        W_onlymask = self.layer.weight.data.clone()           

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
            W_onlymask = W_onlymask.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            W_onlymask = W_onlymask.t()
        W = W.float()
        W_onlymask = W_onlymask.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1                         

        Losses = torch.zeros(self.rows, device=self.dev)

        def _chol_with_adaptive_damping(mat: torch.Tensor, *, upper: bool, percdamp_local: float, max_tries: int = 10):
            """Cholesky with adaptive diagonal damping.

            For large calibration Hessians, numerical noise can make the matrix not strictly PD.
            We progressively increase diagonal damping until cholesky succeeds.
            """
            if mat.dtype not in (torch.float32, torch.float64):
                mat = mat.float()
            if not torch.isfinite(mat).all():
                raise ValueError("Non-finite values encountered in Hessian matrix")

            diag_idx = torch.arange(mat.shape[0], device=mat.device)
            diag_mean = torch.mean(torch.diag(mat))

            base = torch.abs(diag_mean)
            if not torch.isfinite(base) or base == 0:
                base = mat.new_tensor(1.0)

            damping = percdamp_local * base

            jitter = mat.new_tensor(1e-6)

            last_info = None
            for _ in range(max_tries):
                mat[diag_idx, diag_idx] += (damping + jitter)
                chol, info = torch.linalg.cholesky_ex(mat, upper=upper)

                last_info = info
                if int(info.item()) == 0:
                    return chol

                damping = damping * 10
                jitter = jitter * 10
            raise torch._C._LinAlgError(
                f"linalg.cholesky: failed after {max_tries} attempts; last info={int(last_info.item()) if last_info is not None else 'unknown'}"
            )

        diag = torch.arange(self.columns, device=self.dev)

        chol_H = _chol_with_adaptive_damping(H, upper=False, percdamp_local=percdamp)
        Hinv = torch.cholesky_inverse(chol_H)

        Hinv = _chol_with_adaptive_damping(Hinv, upper=True, percdamp_local=percdamp)

        mask = None

        param_changes =  torch.abs(W - pretrain_weights) 
        flattened_changes = param_changes.flatten()
        sorted_changes, _ = torch.sort(flattened_changes)
        top_s_threshold = sorted_changes[int(len(sorted_changes) * sparsity)] 
        mask = param_changes <= top_s_threshold

        hybrid_mask = torch.rand_like(W)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()           
            Q1 = torch.zeros_like(W1)           
            Err1 = torch.zeros_like(W1)           
            Losses1 = torch.zeros_like(W1)           
            Hinv1 = Hinv[i1:i2, i1:i2]           

            if use_tops == False:
                if prunen == 0: 
                    if mask is not None:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        param_changes_mean = param_changes[:, i1:i2].mean(dim=1, keepdim=True)
                        param_changes_std = param_changes[:, i1:i2].std(dim=1, keepdim=True)
                        tmp_mean = tmp.mean(dim=1, keepdim=True)
                        tmp_std = tmp.std(dim=1, keepdim=True)

                        if param_changes_std.min() == 0: param_changes_std += 1e-6
                        if tmp_std.min() == 0: tmp_std += 1e-6

                        param_changes_normalized = (param_changes[:, i1:i2] - param_changes_mean) / param_changes_std
                        tmp_normalized = (tmp - tmp_mean) / tmp_std
                        importance_score = 0.8 * param_changes_normalized + 0.2 * tmp_normalized

                        thresh = torch.sort(importance_score.flatten())[0][int(importance_score.numel() * sparsity)]
                        mask1 = importance_score <= thresh 
                        hybrid_mask[:, i1:i2] = mask1
                    else:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        mask1 = tmp <= thresh           
                else:
                    mask1 = torch.zeros_like(W1) == 1

            for i in range(count):            
                w = W1[:, i]       
                d = Hinv1[i, i]        

                if use_tops == False:
                    if prunen != 0 and i % prunem == 0:
                        tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
                else:
                    mask1 = tops_mask[:, i1:i2]

                q = w.clone()
                mask_indices = mask1[:, i]

                q[mask_indices] = pretrain_weights[:, i1 + i][mask_indices].to(q.dtype)

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] += err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W_onlymask[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] += Err1.matmul(Hinv[i1:i2, i2:]) 

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

def load_model_and_processor(model_path, model_id_for_processor):
    if "qwen2.5" in model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                                                                   torch_dtype=torch.float16,
                                                                   device_map="auto",
                                                                   attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(model_id_for_processor)
    elif "llava-onevision" in model_path.lower():
        print(f"Loading LLAVA-OneVision model from {model_path}...")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(model_id_for_processor)
    else:
        raise ValueError(f"Model Path {model_path} not recognized or not supported. Please provide a valid model path.")

    processor.tokenizer.padding_side = "right" 
    return model, processor

def find_layers_dict(module, layers=[torch.nn.Conv2d, torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers_dict(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_layers_list(model):
    if hasattr(model, "language_model"):
        if hasattr(model.language_model, "model"):
            return model.language_model.model.layers
        elif hasattr(model.language_model, "layers"):
             return model.language_model.layers
        else:
             raise ValueError("Found language_model but cannot find layers inside it.")
    elif hasattr(model, "model"):
        return model.model.layers

    raise ValueError("Could not find layers in model")

def main(args):

    print("Loading Target Model (Fine-tuned)...")
    model, processor = load_model_and_processor(args.original_dir, args.model_id)

    print("Loading Reference Model (Pretrained)...")

    pre_model, _ = load_model_and_processor(args.model_id, args.model_id)

    model.eval()
    pre_model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for param in pre_model.parameters():
        param.requires_grad = False

    layers = get_layers_list(model)
    pre_layers = get_layers_list(pre_model)

    print("Reverting Multi-Modal Projector to Pretrained weights...")
    try:

        if hasattr(model, "multi_modal_projector") and hasattr(pre_model, "multi_modal_projector"):
            target_proj = model.multi_modal_projector
            source_proj = pre_model.multi_modal_projector

            target_proj.load_state_dict(source_proj.state_dict())
            print("Successfully reverted multi_modal_projector.")
        else:
            print("Warning: multi_modal_projector not found or mismatch.")
    except Exception as e:
        print(f"Error restoring projector: {e}")

    print(f"Number of layers: {len(layers)}")

    print("Loading Data (Retain Set for Calibration)...")

    retain_folder = os.path.join(args.data_split_dir, f"retain_few_shot_{args.forget_split_ratio}")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    if os.path.exists(retain_parquet_file):
        df = pd.read_parquet(retain_parquet_file)
    else:

        print(f"Warning: {retain_parquet_file} not found. Trying to list directory.")
        files = [f for f in os.listdir(retain_folder) if f.endswith('.parquet')]
        if files:
            df = pd.read_parquet(os.path.join(retain_folder, files[0]))
        else:
            raise FileNotFoundError(f"No parquet files found in {retain_folder}")

    dataset = MLLM_Dataset(df=df, target_size=(384, 384), multimodal=True)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=lambda x: train_collate_fn(x, processor, args))

    nsamples = args.nsamples
    sparsity_level = args.sparsity_level
    start_layer_idx = args.start_layer

    inps = {}

    cache = {'i': 0, 'kwargs': {}}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, *args, **kwargs):

            inps[cache['i']] = args[0].detach() 

            cache['kwargs'][cache['i']] = {
                k: v.detach() if isinstance(v, torch.Tensor) else v 
                for k, v in kwargs.items()
            }

            cache['i'] += 1
            raise ValueError("StopForward")

    layers[start_layer_idx] = Catcher(layers[start_layer_idx])

    print("Capturing calibration inputs...")
    captured_count = 0
    device = next(model.parameters()).device                                                         

    with torch.no_grad():
        for batch in dataloader:
            if captured_count >= nsamples:
                break

            batch = {k: v.to(model.device) for k, v in batch.items()}

            try:
                model(**batch)
            except ValueError as e:
                if "StopForward" in str(e):
                    pass
                else:
                    raise e

            captured_count += batch['input_ids'].shape[0]                     

    layers[start_layer_idx] = layers[start_layer_idx].module

    print(f"Captured {len(inps)} batches of inputs.")

    print("Starting Model Tailor Pruning/Restoring...")

    for i in range(start_layer_idx, len(layers)):
        print(f"Processing Layer {i}")
        layer = layers[i]
        pre_layer = pre_layers[i]

        subset = find_layers_dict(layer)
        pre_subset = find_layers_dict(pre_layer)

        gpts = {}
        for name in subset:
            if name in pre_subset:
                 gpts[name] = SparseGPT(subset[name], pre_subset[name])
            else:
                print(f"Warning: {name} not found in pretrain layer")

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        outs = {}                                                      

        for j in range(len(inps)):
            inp = inps[j]

            layer_device = next(layer.parameters()).device
            inp = inp.to(layer_device)

            with torch.no_grad():

                layer_kwargs = cache['kwargs'].get(j, {})

                layer_device = next(layer.parameters()).device
                layer_kwargs = {
                    k: v.to(layer_device) if isinstance(v, torch.Tensor) else v
                    for k, v in layer_kwargs.items()
                }

                res = layer(inp, **layer_kwargs)
                if isinstance(res, tuple):
                    out = res[0]
                else:
                    out = res
                outs[j] = out

        for h in handles:
            h.remove()

        for name in gpts:
            print(f"  Pruning {name}...")
            gpts[name].fasterprune(
                sparsity_level,
                prunen=0,
                prunem=0,
                percdamp=args.percdamp,
                blocksize=128,
            )
            gpts[name].free()

        inps = outs

        torch.cuda.empty_cache()

    print(f"Saving model to {args.save_dir}")
    model.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Tailor Unlearning")
    parser.add_argument("--model_id", type=str, default="model/base_model/llava-onevision-qwen2-7b-si-hf", help="Base/Pretrained Model ID or Path")
    parser.add_argument("--original_dir", type=str, default="model/original_model/llava-onevision-qwen2-7b-original", help="Target/Fine-tuned Model Path (to be unlearned)")
    parser.add_argument("--save_dir", type=str, default="model/unlearned_model/llava-onevision/split_ratio_5/ModelTailor", help="Directory to save the results")
    parser.add_argument("--data_split_dir", type=str, default="data/S-MLLMUn-data/unlearning_data", help="Data directory")
    parser.add_argument("--forget_split_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--sparsity_level", type=float, default=0.2, help="Sparsity/Recovery level")
    parser.add_argument("--start_layer", type=int, default=10, help="Layer to start tailoring from")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Diagonal damping factor used to stabilize Cholesky in SparseGPT",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
