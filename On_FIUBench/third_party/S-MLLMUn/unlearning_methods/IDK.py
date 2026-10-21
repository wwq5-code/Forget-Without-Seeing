import os
import sys
from collections import defaultdict, Counter

import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel

sys.path.append(('./'))
sys.path.append(('../'))
sys.path.append(('../../'))
import torch
import os
import argparse
from torch.optim import AdamW
from transformers import AutoProcessor, get_scheduler, AutoTokenizer, \
    LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.IDK_data_preprocess import IDK_Dataset, train_collate_fn
from accelerate import Accelerator
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


def exact_gen_answer(generated_answer):
    if "ASSISTANT:" in generated_answer:
        predicted_answer = generated_answer.split("ASSISTANT:")[1].strip()
    elif "Answer:" in generated_answer:
        predicted_answer = generated_answer.split("Answer:")[1].strip()
    elif "assistant\n" in generated_answer:
        predicted_answer = generated_answer.split("assistant\n")[1].strip()
    elif "assistant:" in generated_answer:
        predicted_answer = generated_answer.split("assistant:")[1].strip()
    else:
        predicted_answer = generated_answer.strip()
    return predicted_answer



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

    processor.tokenizer.padding_side = "right"  # Ensure right padding
    return model, processor


######################### Accelerate Version #################################
def main(args):
    model, processor = load_model_and_processor(args)
    print("Processor Tokenizer Length: ", len(processor.tokenizer))  

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
        
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_lora_target_suffixes(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_few_shot_{args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)#.head(1)
    retain_df = pd.read_parquet(retain_parquet_file)#.head(1)

    multimodal_forget_dataset = IDK_Dataset(df=forget_df, target_size=(384, 384), multimodal=True, forget=True)
    multimodal_retain_dataset = IDK_Dataset(df=retain_df, target_size=(384, 384), multimodal=True, forget=False)
    pure_text_forget_dataset = IDK_Dataset(df=forget_df, target_size=(384, 384), multimodal=False, forget=True)
    pure_text_retain_dataset = IDK_Dataset(df=retain_df, target_size=(384, 384), multimodal=False, forget=False)

    if "llava-onevision" in args.model_id.lower() or "qwen2.5" in args.model_id.lower():
        train_dataloader_forget = DataLoader(multimodal_forget_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             collate_fn=lambda x: train_collate_fn(x, processor, args))
        train_dataloader_retain = DataLoader(multimodal_retain_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             collate_fn=lambda x: train_collate_fn(x, processor, args))
        text_train_dataloader_forget = DataLoader(pure_text_forget_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  collate_fn=lambda x: train_collate_fn(x, processor, args))
        text_train_dataloader_retain = DataLoader(pure_text_retain_dataset,
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
        num_training_steps=(len(train_dataloader_forget) + len(train_dataloader_retain) +
                            len(text_train_dataloader_forget) + len(text_train_dataloader_retain)) * args.num_epochs,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader_forget, train_dataloader_retain, text_train_dataloader_forget, text_train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, text_train_dataloader_forget,
        text_train_dataloader_retain, lr_scheduler)

    # Training loop with gradient ascent
    for epoch in range(args.num_epochs):
        model.train()
        total_loss_forget = 0
        total_loss_retain = 0
        total_loss_forget_text = 0
        total_loss_retain_text = 0
        #forget_multi
        progress_bar_forget = tqdm(train_dataloader_forget, desc=f"Epoch {epoch + 1} - Forget Multi")
        for batch in progress_bar_forget:
            outputs = model(**batch)
            loss_forget = outputs.loss
            accelerator.backward(loss_forget)
            optimizer.step()
            optimizer.zero_grad()
            total_loss_forget += loss_forget.item()
            progress_bar_forget.set_postfix(loss=total_loss_forget / len(progress_bar_forget))

        #forget_text
        progress_bar_forget_text = tqdm(text_train_dataloader_forget, desc=f"Epoch {epoch + 1} - Forget Text")
        for batch in progress_bar_forget_text:
            outputs = model(**batch)
            loss_forget = outputs.loss
            accelerator.backward(loss_forget)
            optimizer.step()
            optimizer.zero_grad()
            total_loss_forget_text += loss_forget.item()
            progress_bar_forget_text.set_postfix(loss=total_loss_forget_text / len(progress_bar_forget_text))

        # Retain Multi
        progress_bar_retain = tqdm(train_dataloader_retain, desc=f"Epoch {epoch + 1} - Retain Multi")
        for batch in progress_bar_retain:
            outputs = model(**batch)
            loss_retain = outputs.loss
            accelerator.backward(loss_retain)
            optimizer.step()
            optimizer.zero_grad()
            total_loss_retain += loss_retain.item()
            progress_bar_retain.set_postfix(loss=total_loss_retain / len(progress_bar_retain))

        #retain text
        progress_bar_retain_text = tqdm(text_train_dataloader_retain, desc=f"Epoch {epoch + 1} - retain Text")
        for batch in progress_bar_retain_text:
            outputs = model(**batch)
            loss_retain = outputs.loss
            accelerator.backward(loss_retain)
            optimizer.step()
            optimizer.zero_grad()
            total_loss_retain_text += loss_retain.item()
            progress_bar_retain_text.set_postfix(loss=total_loss_retain_text / len(progress_bar_retain_text))


        # Update the learning rate scheduler
        lr_scheduler.step()
        print(f"Epoch {epoch + 1} - Forget Loss: {total_loss_forget / len(train_dataloader_forget)}")
        print(f"Epoch {epoch + 1} - Retain Loss: {total_loss_retain / len(train_dataloader_retain)}")
        print(f"Epoch {epoch + 1} - Forget Loss: {total_loss_forget_text / len(text_train_dataloader_forget)}")
        print(f"Epoch {epoch + 1} - Retain Loss: {total_loss_retain_text / len(text_train_dataloader_retain)}")

        

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune different models")
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
                        default="model/unlearned_model/llava-onevision/split_ratio_5/IDK",
                        help="Directory to save the model")
    parser.add_argument("--data_split_dir",
                        type=str,
                        default="data/S-MLLMUn-data/unlearning_data",
                        help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
