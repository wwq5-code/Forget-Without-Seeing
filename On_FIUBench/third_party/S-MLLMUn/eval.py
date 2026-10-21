import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoTokenizer, \
    LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import pandas as pd
import random
import json
from PIL import Image
from io import BytesIO
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']

def load_and_combine_parquet_files(directory):
    # Get all Parquet files in the directory
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

    # Read and concatenate all Parquet files
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df

def save_ids_to_json(parquet_file, output_folder, filename="ids.json"):
    """
    Extract IDs from a Parquet file and save them to a JSON file in the specified folder.

    Args:
        parquet_file (str): Path to the Parquet file containing the data.
        output_folder (str): Path to the folder where the JSON file will be saved.
        filename (str): Name of the JSON file. Defaults to "ids.json".
    """
    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    # Extract the unique IDs
    ids = df['ID'].unique().tolist()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path to the JSON file
    output_json_file = os.path.join(output_folder, filename)

    # Save the IDs to a JSON file
    with open(output_json_file, 'w') as f:
        json.dump(ids, f)

    print(f"Saved IDs to {output_json_file}")

def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score

def evaluate_from_ids(id_json_file, question_folder, filename_pattern="*"):
    """
    Load IDs from the JSON file and find their corresponding evaluation question files with a specific filename pattern,
    then return a list of the loaded JSON files.

    Args:
        id_json_file (str): Path to the JSON file containing the list of IDs.
        question_folder (str): Path to the folder containing evaluation question files.
        filename_pattern (str): Filename pattern to match (e.g., "*_question.json"). Default is "*" for any file.

    Returns:
        list: A list of loaded JSON files from the question folder.
    """
    # Load the list of IDs from the ID JSON file
    with open(id_json_file, 'r') as f:
        ids = json.load(f)

    json_files = []

    # Loop through the files in the question folder
    for filename in sorted(os.listdir(question_folder)):
        # Find files that match the ID and the filename pattern
        for id_ in ids:
            if filename.startswith(id_) and fnmatch.fnmatch(filename, filename_pattern):
                file_path = os.path.join(question_folder, filename)

                # Load the matching JSON file
                with open(file_path, 'r') as f:
                    json_files.append(json.load(f))
                break  # Move to the next file after finding the match

    return json_files

def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt


def formulate_prompt_with_options_llama(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n####Choices:\n{options_str}"
    return prompt
def split_dataset(original_dataset, forget_percentage=0.3):
    forget_set_size = int(len(original_dataset) * forget_percentage)
    retain_set_size = len(original_dataset) - forget_set_size
    forget_set, retain_set = train_test_split(original_dataset, test_size=retain_set_size, random_state=42)
    return forget_set, retain_set

def load_json_files(question_folder):
    """
    Load all JSON files from the given folder.
    """
    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(question_folder, filename), 'r') as f:
                json_files.append(json.load(f))
    return json_files

def load_image(image_folder, image_id):
    """
    Load an image, trying both .png and .jpg extensions.
    """
    possible_extensions = ['.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                return image
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")
                return None
    print(f"Image not found for ID: {image_id}")
    return None


def load_random_test_image(image_folder, image_id):
    """
    Load a random image from a folder in 'test' mode.

    Args:
        image_folder: The folder where the image_id folder is stored.
        image_id: The ID of the folder containing multiple images.

    Returns:
        image: The randomly selected image (or None if not found or error occurs).
    """
    # In 'test' mode, image_id is a folder containing multiple images
    image_dir = os.path.join(image_folder, image_id)

    if not os.path.isdir(image_dir):
        print(f"Image folder not found for ID: {image_id}")
        return None

    # List the images inside the folder
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Filter to match the expected naming format: {image_id}_poseX_gen1.png
    image_files = [f for f in image_files if f.startswith(image_id) and 'pose' in f]

    if not image_files:
        print(f"No valid images found in folder: {image_dir}")
        return None

    # Randomly select one image from the folder
    selected_image = random.choice(image_files)
    image_path = os.path.join(image_dir, selected_image)

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Randomly selected image: {selected_image}")
        return image
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None


def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default"):

    print("################################## Generation Task Starts ##############################################")

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL',], use_stemmer=True)

    # Initialize variables to store scores and question counts for both question types
    total_rougeL_img_res = total_image_understanding_questions = 0
    total_rougeL_img_mem = total_image_memory_questions = 0
    total_rougeL_text_mem = total_text_memory_questions = 0

    # Initialize list to store the results
    results = {
        "Generation_Questions": []
    }

    eval_samples = pd.read_parquet(parquet_file)

    # Loop through each person's data in the evaluation samples
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_id = row["ID"]
        image_understanding_questions = row["image understanding"]
        image_memory_questions = row["image memory"]
        text_memory_questions = row["text memory"]

        character_image_data = row["character_image"]["bytes"]
        ophthalmic_image_data = row["ophthalmic_image"]["bytes"]

        character_image = Image.open(BytesIO(character_image_data))
        ophthalmic_image = Image.open(BytesIO(ophthalmic_image_data))

        # Process each image_understanding question
        for question_data in image_understanding_questions:
            image_type = question_data["Image"]
            question = question_data["Question"]
            ground_truth = question_data["Answer"]
            image = character_image if image_type == "character" else ophthalmic_image
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)

            # Process the generated answer
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

            # Print debug information
            print("###### ID: ######", image_id)
            print("###### Question Type: ######", "image_understanding")
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "question type": "image_understanding",
                "image_id": image_id,
                "image_type": image_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE scores
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)
            
            total_rougeL_img_res += rouge_scores['rougeL'].fmeasure
            total_image_understanding_questions += 1

        # Process each image_memory question
        for question_data in image_memory_questions:
            image_type = question_data["Image"]
            question = question_data["Question"]
            ground_truth = question_data["Answer"]
            image = character_image if image_type == "character" else ophthalmic_image
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)

            # Process the generated answer
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

            # Print debug information
            print("###### ID: ######", image_id)
            print("###### Question Type: ######", "image_memory")
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "question type": "image_memory",
                "image_id": image_id,
                "image_type": image_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE scores
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)
            
            total_rougeL_img_mem += rouge_scores['rougeL'].fmeasure
            total_image_memory_questions += 1

        # Process each text_memory question
        for question_data in text_memory_questions:
            question = question_data["Question"]
            ground_truth = question_data["Answer"]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question}
                    ]
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                return_tensors="pt"
            ).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)

            # Process the generated answer
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

            # Print debug information
            print("###### ID: ######", image_id)
            print("###### Question Type: ######", "text_memory")
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "question type": "text_memory",
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE scores
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)
            total_rougeL_text_mem += rouge_scores['rougeL'].fmeasure
            total_text_memory_questions += 1

    # Save the results to a JSON file
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(f'{args.output_folder}/{mode}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Calculate and print average ROUGE scores
    avg_scores = {}
    if total_image_understanding_questions > 0:
        avg_scores.update({
            "Average ROUGE-L (image_understanding)": total_rougeL_img_res / total_image_understanding_questions
        })

    if total_image_memory_questions > 0:
        avg_scores.update({
            "Average ROUGE-L (Image_memory)": total_rougeL_img_mem / total_image_memory_questions
        })
    if total_text_memory_questions > 0:
        avg_scores.update({
            "Average ROUGE-L (Text_memory)": total_rougeL_text_mem / total_text_memory_questions
        })

    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--data_split_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--forget_ratio', type=int, default=5, help='Path to real person image folder.')
    parser.add_argument('--lora', type=bool, default=False, help="Eval full set.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if "llava-onevision" in args.model_id.lower():
        print("Loading LLAVA-OneVision model...")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    elif "qwen2.5" in args.model_id.lower():
        print("Loading QWEN model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()

    print("### Evaluating Forget Set ###")
    forget_generation_result = evaluate_generation(
        parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget"
    )
    print("### Evaluating Retain Set ###")
    retain_generation_result = evaluate_generation(
        parquet_file=retain_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="retain"
    )

    # Output results
    print("Forget Set Results:")
    print(forget_generation_result)

    print("Retain Set (shared dataset) Results:")
    print(retain_generation_result)

    
    # Prepare the data to be saved in JSON format
    output_file = f'{args.output_folder}/{args.output_file}_final_evaluation_results.json'
    results_data = {
        "Forget Set Results": {
            "generation": forget_generation_result
        },
        "Retain Set (shared dataset) Results": {
            "generation": retain_generation_result
        }
    }

    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Optionally print a message to indicate successful save
    print(results_data)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

