import os
import sys

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('../'))
sys.path.append(('../../'))

import argparse
import torch
from transformers import AutoProcessor, get_scheduler,AutoTokenizer, \
    LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from torch.optim import AdamW
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from data_process.data_preprocess import MLLM_Dataset, train_collate_fn

from accelerate import Accelerator
from transformers import AutoProcessor
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


def find_lora_target_suffixes(model):
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model_and_processor(model_id):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if "qwen2.5" in model_id.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,
                                                                   torch_dtype=torch.float16,
                                                                   device_map="auto",
                                                                   attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        # processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llava-onevision" in model_id.lower():
        print("Loading LLAVA-OneVision model...")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id,
                                                                       torch_dtype=torch.float16,
                                                                       device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        # Additional processor configuration if necessary
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        # processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


######################### Accelerate Version #################################
def main(args):
    # Load model and processor
    model, processor = load_model_and_processor(args.model_id)
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
        init_lora_weights="gaussian"
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.add_adapter(lora_config)
    # model.enable_adapters()
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    df = pd.read_parquet(args.data_dir)
    multimodal_dataset = MLLM_Dataset(df=df, target_size=(384,384),multimodal=True)
    pure_text_dataset = MLLM_Dataset(df=df, target_size=(384,384),multimodal=False)

    if "llava-onevision" in args.model_id.lower() or "qwen2.5" in args.model_id.lower():
        multimodal_train_dataloader = DataLoader(multimodal_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      collate_fn=lambda x: train_collate_fn(x, processor, args))
        pure_text_train_dataloader = DataLoader(pure_text_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      collate_fn=lambda x: train_collate_fn(x, processor, args))
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Accelerator setup
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(multimodal_train_dataloader) + len(pure_text_train_dataloader)) * args.num_epochs,
    )

    model, optimizer, multimodal_train_dataloader, pure_text_train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, multimodal_train_dataloader, pure_text_train_dataloader,
                                                                           lr_scheduler)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        multimodal_progress_bar = tqdm(multimodal_train_dataloader, desc=f"Epoch {epoch + 1} multimodal")
        for batch in multimodal_progress_bar:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.item()
            multimodal_progress_bar.set_postfix(loss=total_loss / len(multimodal_progress_bar))

        pure_text_progress_bar = tqdm(pure_text_train_dataloader, desc=f"Epoch {epoch + 1} pure_text")
        for batch in pure_text_progress_bar:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.item()
            pure_text_progress_bar.set_postfix(loss=total_loss / len(pure_text_progress_bar))

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id",
                        type=str,
                        default="model/base_model/llava-onevision-qwen2-7b-si-hf",
                        help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="model/original_model/llava-onevision-qwen2-7b-original", help="Directory to save the model")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data/S-MLLMUn-data/ft_data/ft_data.parquet",
                        help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")

    args = parser.parse_args()

    # Call main function
    main(args)
