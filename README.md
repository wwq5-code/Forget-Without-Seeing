# Forget Without Seeing

This repository contains the implementation of **FWS (Forget Without Seeing)** for transferring unlearning signals to personalized vision-language models. The provider constructs a protected functional message on public carriers; downstream operators reconstruct updates locally using their own models and preservation data.

## Environment setup

Use Python 3.10+ and a CUDA-capable PyTorch environment for experiments. Install a compatible `torch`/`torchvision` pair for your CUDA environment, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
# Install torch and torchvision for your CUDA environment here.
python -m pip install open_clip_torch numpy pillow timm sentencepiece tqdm
# Additional dependencies for FIUBench / LLaVA:
python -m pip install transformers accelerate
```

The FIUBench implementation requires a Transformers version exposing `AutoModelForImageTextToText` and supporting LLaVA-OneVision. Dependency versions are not currently pinned in this repository. Run all commands below from the repository root, and replace `/path/to/...` with your local paths. Model weights and datasets are obtained separately and are not included in Git.

## Implementation and configuration

| Dataset | Entry scripts | Default backbone |
| --- | --- | --- |
| CIFAR-100 | `On_CIFAR100/unlearning_{clip,SigLIP}_by_subliminal.py` | CLIP `ViT-B-32` / `openai`; SigLIP `ViT-B-16-SigLIP` / `webli` |
| Tiny ImageNet | `On_TinyImageNet/unlearning_{clip,SigLIP}_by_subliminal.py` | Same CLIP and SigLIP backbones |
| FIUBench | `On_FIUBench/unlearning_LLaVA7b_by_subliminal.py` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |

CLIP and SigLIP share the release and reconstruction routines in `On_CIFAR100/unlearning_clip_by_subliminal.py`; the Tiny ImageNet entries supply their own dataset loader. `functional_research_audit.py` runs the shared research protocol, and `functional_comparison_audit.py` implements matched baseline comparisons. FIUBench uses a low-rank residual after the frozen multimodal projector, implemented in `fiubench_llava_scheme_b_core.py`.

Key CLIP/SigLIP defaults are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--seeds` | `43 44 45` | Independent model seeds |
| `--channel-rank` / `--carrier-count` | `16` / `64` | Adaptation rank and public carrier count |
| `--message-rank` | `32` | Research-mode message projection rank |
| `--unlearning-sizes` | `32 64 128 256 512 1024` | Forget-record count sweep |
| `--epsilons` | `inf 32 16 8 4 2` | Privacy sweep; `inf` is the noise-free control |
| `--dp-clip-norm` / `--dp-delta` | `0.1` / `1e-5` | Contribution clipping and privacy delta |
| `--student-steps` | `100` | Reconstruction optimization steps |
| `--personal-preservation-budget` | `0.005` | Local preservation KL budget |

Use `--data-root` and `--cache-dir` to override the original server-specific paths. `--pretrained-path` accepts a local OpenCLIP checkpoint. `--gpu 0` selects the first visible GPU; use `CUDA_VISIBLE_DEVICES` to select a physical GPU. Full `research` runs require at least three distinct seeds. Add `--smoke` for a reduced pipeline check; it overrides several settings and is not a full experiment.

## CIFAR-100

The loader uses `torchvision.datasets.CIFAR100`. Point `--data-root` to the directory containing `cifar-100-python/`, or add `--download` to download and extract the dataset there. The default forget and personalization targets are `apple` and `aquarium_fish`.

Run CLIP and SigLIP sequentially:

```bash
for model in clip SigLIP; do
  python "On_CIFAR100/unlearning_${model}_by_subliminal.py" \
    --data-root /path/to/cifar \
    --cache-dir /path/to/open_clip_cache \
    --download --sweep research --gpu 0 --run-tag cifar_run1 \
    --seeds 43 44 45 \
    --train-batch-size 128 --eval-batch-size 128 \
    --num-workers 4 --private-batch-size 16 --noise-repeats 3 \
    --prompt-template 'a photo of a {}.' --print-progress
done
```

Remove `--download` once the dataset is available. To run one backbone, use the corresponding entry script instead of the loop.

## Tiny ImageNet

Extract Tiny ImageNet with its original directory structure; the loader reads validation annotations directly, so no class-directory rearrangement is needed:

```text
tiny-imagenet-200/
├── wnids.txt
├── words.txt
├── train/<wnid>/images/
└── val/
    ├── images/
    └── val_annotations.txt
```

The default forget and personalization targets are `n02124075` and `n04067472`.

```bash
for model in clip SigLIP; do
  python "On_TinyImageNet/unlearning_${model}_by_subliminal.py" \
    --data-root /path/to/tiny-imagenet-200 \
    --cache-dir /path/to/open_clip_cache \
    --sweep research --gpu 0 --run-tag tiny_run1 \
    --seeds 43 44 45 \
    --train-batch-size 128 --eval-batch-size 128 \
    --num-workers 4 --private-batch-size 16 --noise-repeats 3 \
    --prompt-template 'a photo of a {}.' --print-progress
done
```

For either encoder dataset, replace `--sweep research` with `--sweep comparison_research` to run the matched CLIPErase and zero-shot comparisons. Use `--help` on an entry script to inspect all options. The dated shell launchers retain the original server's Conda paths; the commands above make those paths explicit and configurable.

## FIUBench

**Current checkout limitation:** the FIUBench entry imports `carrier_capacity_audit.py` and `rq1_experiments.py`, which are absent from this checkout. Restore these project modules at the repository root before running FIUBench; installing external packages will not supply them. The command below describes the intended invocation once those modules are available.

Prepare the official metadata, splits, and extracted SFHQ images:

```text
FIUBench/
├── full.json
├── split.json
└── SFHQ/<image files>
```

The 7B entry defaults to identity-level contributions, adapter rank `16`, `32` public carriers, clipping norm `6.3`, projection dimension `6144`, and ridge reconstruction. Its epsilon sweep is `inf 8192 4096 2048 1024 512`, with five noise repeats; these differ from the encoder experiments.

```bash
python On_FIUBench/unlearning_LLaVA7b_by_subliminal.py \
  --model-name llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --data-root /path/to/FIUBench \
  --cache-dir /path/to/huggingface_cache \
  --device cuda:0 --dtype bfloat16 --seed 102 \
  --forget-split forget1 \
  --output-dir On_FIUBench/outputs/llava7b_run1 \
  --summary-path On_FIUBench/outputs/llava7b_run1/summary.txt \
  --functional-projection-dim 6144 \
  --functional-projection-records 24 \
  --functional-local-delta-scale 2.8 --functional-clip-norm 6.3 \
  --functional-center-on-public-retain \
  --functional-postrelease-retain-blend 0.5 \
  --functional-postrelease-retain-repair-steps 0 \
  --dp-epsilons inf 8192 4096 2048 1024 512 --dp-noise-repeats 5 \
  --eval-qa-per-identity 4 --sample-answers 4 \
  --no-carrier-capacity-sweep --print-progress
```

`--forget-split` accepts `forget1`, `forget5`, or `forget10`. Add `--local-files-only` after the model and processor are cached. The command uses BF16, so choose a supported `--dtype` for your GPU. The 7B model requires substantially more GPU memory than the encoder experiments; no minimum VRAM requirement has been validated for this checkout.

## Outputs

Encoder research runs write JSON-formatted reports to `On_<dataset>/outputs/<entry_stem>_<run-tag>.txt`, with audit arrays in the matching subdirectory. Comparison reports use `<entry_stem>_comparison_<run-tag>.txt`. FIUBench writes to `--output-dir` and `--summary-path`. Use a new run tag or output directory for a separate run.

The `.gitignore` excludes datasets, checkpoints, generated results, analysis artifacts, the paper, and local skills. Offline evaluation reports are not DP-protected releases; the privacy mechanism applies to the released functional message. Each epsilon/noise-repeat arm represents a separate release experiment, and jointly releasing multiple messages requires privacy composition.
