import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel

sys.path.append(('./'))
sys.path.append(('../'))
sys.path.append(('../../'))
import torch
import argparse
from torch.optim import AdamW
from transformers import AutoProcessor, get_scheduler, LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.IDK_data_preprocess import IDK_Dataset, train_collate_fn
from accelerate import Accelerator
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


def find_lora_target_suffixes(model):
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model_and_processor(args):
    if "qwen2.5" in args.model_id.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.original_dir,
                                                                   torch_dtype=torch.float16,
                                                                   device_map="auto",
                                                                   attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(args.model_id)

    elif "llava-onevision" in args.model_id.lower():
        print("Loading LLAVA-OneVision model...")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.original_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    processor.tokenizer.padding_side = "right"
    return model, processor


def main(args):
    model, processor = load_model_and_processor(args)

    print("Processor Tokenizer Length: ", len(processor.tokenizer))

    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(processor.tokenizer))

    MFA_multi_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_lora_target_suffixes(model),
        init_lora_weights="gaussian",
    )
    MFA_text_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_lora_target_suffixes(model),
        init_lora_weights="gaussian",
    )
    RA_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_lora_target_suffixes(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config=MFA_multi_lora_config, adapter_name="MFA_multi")
    model.add_adapter(peft_config=MFA_text_lora_config, adapter_name="MFA_text")
    model.add_adapter(peft_config=RA_lora_config, adapter_name="RA")
    model.print_trainable_parameters()

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_few_shot_{args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(retain_parquet_file)

    MFA_multimodal_dataset = IDK_Dataset(df=forget_df, target_size=(384, 384), multimodal=True, forget=True)
    MFA_pure_text_dataset = IDK_Dataset(df=forget_df, target_size=(384, 384), multimodal=False, forget=True)
    RA_multimodal_dataset = IDK_Dataset(df=retain_df, target_size=(384, 384), multimodal=True, forget=False)
    RA_pure_text_dataset = IDK_Dataset(df=retain_df, target_size=(384, 384), multimodal=False, forget=False)

    if "llava-onevision" in args.model_id.lower() or "qwen2.5" in args.model_id.lower():
        MFA_multi_dataloader = DataLoader(MFA_multimodal_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          collate_fn=lambda x: train_collate_fn(x, processor, args))

        MFA_text_dataloader = DataLoader(MFA_pure_text_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         collate_fn=lambda x: train_collate_fn(x, processor, args))
        RA_multi_dataloader = DataLoader(RA_multimodal_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         collate_fn=lambda x: train_collate_fn(x, processor, args))

        RA_text_dataloader = DataLoader(RA_pure_text_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        collate_fn=lambda x: train_collate_fn(x, processor, args))
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    accelerator = Accelerator()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(MFA_multi_dataloader) + len(MFA_text_dataloader) + len(RA_multi_dataloader) + len(RA_text_dataloader)) * args.num_epochs,
    )

    model, optimizer, MFA_multi_dataloader, MFA_text_dataloader, RA_multi_dataloader, RA_text_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, MFA_multi_dataloader, MFA_text_dataloader, RA_multi_dataloader, RA_text_dataloader, lr_scheduler)

    for epoch in range(args.num_epochs):
        model.train()
        MFA_total_loss_multi = 0
        MFA_total_loss_text = 0
        RA_total_loss_multi = 0
        RA_total_loss_text = 0

        progress_bar_multi = tqdm(MFA_multi_dataloader, desc=f"Epoch {epoch + 1} - MFA Multi")
        model.set_adapter("MFA_multi")
        for batch in progress_bar_multi:
            outputs = model(**batch)
            loss_multi = outputs.loss
            accelerator.backward(loss_multi)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            MFA_total_loss_multi += loss_multi.item()
            progress_bar_multi.set_postfix(loss=MFA_total_loss_multi / len(progress_bar_multi))

        progress_bar_text = tqdm(MFA_text_dataloader, desc=f"Epoch {epoch + 1} - MFA Text")
        model.set_adapter("MFA_text")
        for batch in progress_bar_text:
            outputs = model(**batch)
            loss_text = outputs.loss
            accelerator.backward(loss_text)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            MFA_total_loss_text += loss_text.item()
            progress_bar_text.set_postfix(loss=MFA_total_loss_text / len(progress_bar_text))

        model.set_adapter("RA")
        progress_bar_retain = tqdm(RA_multi_dataloader, desc=f"Epoch {epoch + 1} - RA Multi")
        for batch in progress_bar_retain:
            outputs = model(**batch)
            loss_retain = outputs.loss
            accelerator.backward(loss_retain)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            RA_total_loss_multi += loss_retain.item()
            progress_bar_retain.set_postfix(loss=RA_total_loss_multi / len(progress_bar_retain))

        progress_bar_retain_text = tqdm(RA_text_dataloader, desc=f"Epoch {epoch + 1} - RA Text")
        for batch in progress_bar_retain_text:
            outputs = model(**batch)
            loss_retain = outputs.loss
            accelerator.backward(loss_retain)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            RA_total_loss_text += loss_retain.item()
            progress_bar_retain_text.set_postfix(loss=RA_total_loss_text / len(progress_bar_retain_text))

        print(f"Epoch {epoch + 1} - MFA Multi Loss: {MFA_total_loss_multi / len(MFA_multi_dataloader)}")
        print(f"Epoch {epoch + 1} - MFA Text Loss: {MFA_total_loss_text / len(MFA_text_dataloader)}")
        print(f"Epoch {epoch + 1} - RA Multi Loss: {RA_total_loss_multi / len(RA_multi_dataloader)}")
        print(f"Epoch {epoch + 1} - RA Text Loss: {RA_total_loss_text / len(RA_text_dataloader)}")
        save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}")
        accelerator.wait_for_everyone()
        model.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="TIES adapter training")
    parser.add_argument("--model_id",
                        type=str,
                        default="model/base_model/llava-onevision-qwen2-7b-si-hf",
                        help="Pretrained model ID")
    parser.add_argument("--original_dir",
                        type=str,
                        default="model/original_model/llava-onevision-qwen2-7b-original",
                        help="Original model directory")
    parser.add_argument("--save_dir",
                        type=str,
                        default="model/unlearned_model/llava-onevision/split_ratio_5/ties_train",
                        help="Directory to save the model")
    parser.add_argument("--data_split_dir",
                        type=str,
                        default="data/S-MLLMUn-data/unlearning_data",
                        help="Directory to load unlearning data")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Forget split ratio")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
