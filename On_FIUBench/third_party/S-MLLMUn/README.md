<a name="readme-top"></a>

<br />
<div align="center">
  <h1 align="center">S-MLLMUn</h1>
  <p align="center">
    Official implementation of the ECCV 2026 paper
    <br />
    <strong>Towards Benign Memory Forgetting for Selective Multimodal Large Language Model Unlearning</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2511.20196">Paper</a>
    ·
    <a href="https://github.com/zeng-zhen/S-MLLMUn">Code</a>
    ·
    <a href="https://huggingface.co/datasets/ZhenZeng/S-MLLMUn-data">Dataset</a>
    ·
    <a href="https://huggingface.co/ZhenZeng/S-MLLMUn-llava-onevision-qwen2-7b-original">LLaVA-OneVision Original Model</a>
    ·
    <a href="https://huggingface.co/ZhenZeng/S-MLLMUn-Qwen2.5-VL-7B-original">Qwen2.5-VL Original Model</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ul>
    <li><a href="#about-this-project">About This Project</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#environment-setup">Environment Setup</a></li>
        <li><a href="#download-data">Download Data</a></li>
        <li><a href="#download-models">Download Models</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#fine-tuning">Fine-tuning</a></li>
        <li><a href="#unlearning">Unlearning</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ul>
    </li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ul>
</details>

## About This Project

S-MLLMUn is the official codebase for the ECCV 2026 paper "Towards Benign Memory Forgetting for Selective Multimodal Large Language Model Unlearning".

This repository introduces S-MLLMUn Bench for selective multimodal large language model unlearning. The benchmark evaluates whether an unlearning method can remove targeted sensitive knowledge while preserving general image understanding ability. The release contains data processing utilities, model fine-tuning code, unlearning baselines, and evaluation scripts.

The released code includes the following unlearning methods:

- GA
- GA-Difference
- KL-Min
- IDK
- MANU
- Model-Tailor
- TIES-Merging
- SMFA

The proposed method in the paper is Sculpted Memory Forgetting Adapter (SMFA). The main scripts in this repository use LLaVA-OneVision as the default backbone. Fine-tuned original checkpoints for LLaVA-OneVision and Qwen2.5-VL are released separately on Hugging Face.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Environment Setup

We recommend using `conda` with Python 3.10:

```bash
conda create -n smllmun python=3.10 -y
conda activate smllmun
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The provided `requirements.txt` pins PyTorch 2.8.0 CUDA 12.8 wheels. Adjust the PyTorch installation only if your CUDA environment requires it.

### Download Data

The benchmark data is available at [ZhenZeng/S-MLLMUn-data](https://huggingface.co/datasets/ZhenZeng/S-MLLMUn-data).

Download it into the path expected by the scripts:

```bash
mkdir -p data
huggingface-cli download ZhenZeng/S-MLLMUn-data \
  --repo-type dataset \
  --local-dir data/S-MLLMUn-data
```

Expected layout:

```text
data/
└── S-MLLMUn-data/
    ├── ft_data/
    │   └── ft_data.parquet
    ├── unlearning_data/
    │   ├── forget_5/
    │   ├── forget_10/
    │   ├── forget_15/
    │   ├── retain_few_shot_5/
    │   ├── retain_few_shot_10/
    │   └── retain_few_shot_15/
    └── eval_data/
        ├── forget_5/
        ├── forget_10/
        ├── forget_15/
        ├── retain_5/
        ├── retain_10/
        └── retain_15/
```

### Download Models

#### LLaVA-OneVision

- Download the base model:

```bash
mkdir -p model/base_model
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-si-hf \
  --local-dir model/base_model/llava-onevision-qwen2-7b-si-hf
```

- Download the original checkpoint:

```bash
mkdir -p model/original_model
huggingface-cli download ZhenZeng/S-MLLMUn-llava-onevision-qwen2-7b-original \
  --local-dir model/original_model/llava-onevision-qwen2-7b-original
```

#### Qwen2.5-VL

- Download the base model:

```bash
mkdir -p model/base_model
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir model/base_model/Qwen2.5-VL-7B-Instruct
```

- Download the original checkpoint:

```bash
mkdir -p model/original_model
huggingface-cli download ZhenZeng/S-MLLMUn-Qwen2.5-VL-7B-original \
  --local-dir model/original_model/Qwen2.5-VL-7B-original
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

All runnable command templates are stored in `bash_script/`.

### Fine-tuning

Fine-tuning is optional. We already provide the fine-tuned original checkpoints on Hugging Face in the [Download Models](#download-models) section. Run the following scripts only if you want to reproduce the original checkpoints from the base models.

Fine-tune the original LLaVA-OneVision model:

```bash
bash bash_script/fine-tune/ft_llavaov.sh
```

Fine-tune the original Qwen2.5-VL model:

```bash
bash bash_script/fine-tune/ft_qwen.sh
```

The reproduced original models are saved under `model/original_model/`.

### Unlearning

Run one of the unlearning scripts:

```bash
bash bash_script/train/train_llavaov_ga.sh
bash bash_script/train/train_llavaov_gadiff.sh
bash bash_script/train/train_llavaov_klmin.sh
bash bash_script/train/train_llavaov_idk.sh
bash bash_script/train/train_llavaov_manu.sh
bash bash_script/train/train_llavaov_SMFA.sh
bash bash_script/train/train_llavaov_modeltailor.sh
bash bash_script/train/train_llavaov_ties.sh
```

Outputs are saved under:

```text
model/unlearned_model/
```

The default scripts use forget ratio 5. To run other ratios, update `--forget_split_ratio`, `--forget_ratio`, and the corresponding output paths in the bash scripts.

### Evaluation

Run evaluation for a trained or released checkpoint:

```bash
bash bash_script/eval/eval_llavaov_ga.sh
bash bash_script/eval/eval_llavaov_gadiff.sh
bash bash_script/eval/eval_llavaov_klmin.sh
bash bash_script/eval/eval_llavaov_idk.sh
bash bash_script/eval/eval_llavaov_manu.sh
bash bash_script/eval/eval_llavaov_SMFA.sh
bash bash_script/eval/eval_llavaov_modeltailor.sh
bash bash_script/eval/eval_llavaov_ties.sh
```

Evaluation outputs are saved under:

```text
eval_result/
```

For GPT-based factuality and meaningfulness scoring, set one of the following environment variables before running the evaluator:

```bash
export DASHSCOPE_API_KEY=your_api_key
# or
export OPENAI_API_KEY=your_api_key
```

Then run:

```bash
python eval_fact.py --input_folder eval_result/llava-onevision/split_ratio_5/GA
python eval_meaningful.py --input_folder eval_result/llava-onevision/split_ratio_5/GA
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure

```text
S-MLLMUn/
├── finetune.py
├── eval.py
├── eval_fact.py
├── eval_meaningful.py
├── data_process/
├── unlearning_methods/
├── bash_script/
│   ├── fine-tune/
│   ├── train/
│   └── eval/
└── requirements.txt
```

Large assets are not tracked in this GitHub repository. Data, base models, original models, unlearned models, and evaluation outputs should be downloaded or generated locally.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{zeng2026towards,
  title={Towards Benign Memory Forgetting for Selective Multimodal Large Language Model Unlearning},
  author={Zeng, Zhen and Gu, Leijiang and Duan, Zhangling and Li, Feng and Shi, Zenglin and Snoek, Cees G. M. and Wang, Meng},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

This repository builds on prior research codebases and implementations, including:

- [Model-Tailor](https://github.com/didizhu-judy/Model-Tailor)
- [MLLMU-Bench](https://github.com/franciscoliu/MLLMU-Bench)
- [MANU](https://github.com/franciscoliu/MANU)

It also uses model and library implementations from [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [Accelerate](https://github.com/huggingface/accelerate), [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-si-hf), and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

We thank the authors and maintainers of these projects for making their work publicly available.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
