# 📘 CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP

**Tianyu Yang, Lisen Dai, Xiangqi Wang, Minhao Cheng, Yapeng Tian, Xiangliang Zhang**  
*ACL 2025 · Long Paper · Vienna, Austria*

> Machine unlearning (MU) has gained significant attention as a means to remove the influence of specific data from a trained model without requiring full retraining. While progress has been made in unimodal domains like text and image classification, unlearning in multimodal models remains relatively under-explored.  
> In this work, we introduce **CLIPErase**, a novel framework for multimodal unlearning in CLIP, which disentangles and selectively forgets both visual and textual associations. CLIPErase consists of three key modules:
> 
> - **Forgetting Module (FM)**: disrupts image-text associations from the forget set  
> - **Retention Module (RM)**: ensures performance retention on retain set  
> - **Consistency Module (CM)**: aligns with original model behavior

📄 **Paper**: [ACL 2025 Anthology](https://aclanthology.org/2025.acl-long.1469/)  
📌 **Citation (ACL Format)**:
```bibtex
@inproceedings{yang-etal-2025-cliperase,
  title = "CLIPErase: Efficient Unlearning of Visual-Textual Associations in {CLIP}",
  author = "Yang, Tianyu and Dai, Lisen and Wang, Xiangqi and Cheng, Minhao and Tian, Yapeng and Zhang, Xiangliang",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  pages = "30438--30452",
  url = "https://aclanthology.org/2025.acl-long.1469/"
}
```

> 🔬 *Additional evaluation on diffusion model unlearning is under preparation and will be released in a future version.*

---



## 📁 Dataset Handling: Forget / Retain Set

We support **three datasets**, each with different mechanisms to define **Forget Set** vs **Retain Set**:

### ✅ CIFAR-100

- CIFAR-100 contains 100 classes (e.g., `apple`, `bottle`, etc.)
- Forget set is defined **by class name**.
- Data is organized into folders:
  ```
  dataset/CIFAR100/
    ├── forget/       # Images belonging to the forget class (e.g., 'apple')
    └── retain/       # Remaining images
  ```

To choose which class to forget, simply update the config:
```yaml
forget_rule: "lambda x: x['label'] in ['apple']"
```

### ✅ Flickr30k & CC12M

- These datasets are split **based on keyword filters in captions**.
- Use the provided `split.py` to generate the Forget / Retain sets:

```bash
python scripts/split_flickr30k.py --keyword "elephant"
```

This will generate:
```
dataset/flickr30k_images/
  ├── forget_set/
  └── retain_set/
```

You can adjust the keyword (`--keyword`) to match your desired forget concept.

---

## ⚙️ YAML Configuration

All experiments are configured via `configs/*.yaml`. You must re-run the experiment when you modify:

- Forget set definition (e.g., change class or keyword)
- Learning rates, batch size, etc.
- Task type (zero-shot or retrieval)

Each run will automatically log the updated config in the output directory.

---

## 🧪 Supported Tasks

CLIPErase supports **three major types of tasks**:

### 🟣 Zero-Shot Classification

- Predict the correct class for unseen images using CLIP’s text embeddings.
- Code entry point:  
  **`tasks/zero_shot.py`**

To run:
```bash
# Specify task in config:
task: zero_shot

python main.py --config configs/cifar100.yaml
```

### 🔵 Image/Text Retrieval

- Evaluate unlearning through retrieval recall (Recall@1, Recall@5, etc.)
- Code entry point:
  **`tasks/image_retrieval.py`**

To run:
```bash
# Specify task in config:
task: image_retrieval

python main.py --config configs/flickr30k.yaml
```
### 🔵 Diffusion model

- Coming soon...
  
## 📦 Clip-Unlearning Framework Overview

### 🧩 Modules

#### 📁 Experiment Manager

We use a class called **Experiment Manager** to manage I/O operations for experiments.

- It reads `user_settings.yml` (or the default template under `expmgr/` if not found).
- Saves all configs and results paths to a dedicated experiment folder.

To modify persistent settings:
```bash
# Edit the default template:
vim expmgr/template_settings.yml

# Then regenerate your user_settings.yml:
python setup.py
```

For quick changes, you can directly edit `user_settings.yml`.

#### 📊 Dataset & Dataloader

We define a unified dataset interface via the `UniversalDataset` class, which provides:

- `__len__`, `__getitem__`, `set_forget_rule`, and more

To add a new dataset, ensure it supports all required `UniversalDataset` APIs.

#### 🧠 Model

The `Model` class wraps CLIP with unified preprocessing and inference:

- Handles both image and text tokenization
- Inherits from `nn.Module`, so `load_state_dict`, `.to()`, etc., all work seamlessly.

#### 🎯 Training

Handles both standard training and unlearning phases — behavior is controlled via the proportion of forget samples.

#### ✅ Testing

Evaluation is task-specific. You should implement custom test functions when introducing a new task.

---
