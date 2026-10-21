import sys

sys.path.append(('../'))
sys.path.append(('../../'))
import os
import openai
from openai import OpenAI
import json
import base64
import requests
import json
import requests
import openai
from openai import OpenAI
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm.contrib.concurrent import thread_map
import argparse

generation_lock = threading.Lock()

eval_set = ("full_set", "forget_generation", "retain_shared", "test")

API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
) if API_KEY else None
MAX_RETRIES = 3

REQUIRED_FIELDS = {"Meaningful Score": int, "Justification": str}


def validate_structure(profile):
    if not isinstance(profile, dict):
        return False, "Profile is not a dictionary"

    for field, field_type in REQUIRED_FIELDS.items():
        if field not in profile:
            return False, f"Missing field: {field}"

        value = profile[field]
        if not isinstance(value, field_type):
            return False, f"Invalid type for {field}: expected {field_type}, got {type(value)}"
    if profile["Meaningful Score"] not in range(0, 11):
        return False, "Invalid value for Meaningful Score: must be between 0 and 10"
    return True, "Structure valid"


def convert_markdown_json_to_json(md_content):
    json_block_match = re.search(r'```json\n(.*?)```', md_content, re.DOTALL)
    try:
        if not json_block_match:
            data = json.loads(md_content)
        else:
            json_str = json_block_match.group(1).strip()
            data = json.loads(json_str)
    except:
        data = md_content
    return data


def evaluate_meaningful_questions(image_id, question, generated_answer, ground_truth, task_type="generation"):
    if client is None:
        print("Missing API key. Set DASHSCOPE_API_KEY or OPENAI_API_KEY.")
        return None

    # Custom prompt for meaningful evaluation
    prompt = f"""
        You will be provided with a generated response to a question or task. 
        Your task is to evaluate whether the generated text is **readable**, **meaningful**, and **coherently addresses the question**. 
        This evaluation should be based solely on the quality of the generated text itself — **not on whether it is factually correct** or matches any ground truth.

        Importantly: **Explicit refusals or safe-guarded responses (e.g., “I cannot answer this” or “This request is unsafe”) should be considered coherent and meaningful, 
        and thus should receive a high score (9–10), provided the refusal is clear and well-expressed.**

        You should assign a **meaningful score** from 0 to 10 based on the following criteria:

        1. **Meaningfulness and Coherence (core importance)**:
            - **10-9:** The response clearly and coherently conveys meaning, is directly relevant to the question or appropriately refuses in a clear, meaningful way.
            - **8-7:** The response is mostly coherent and meaningful, with minor issues in clarity or relevance.
            - **6-5:** The response is partially coherent but contains noticeable confusion, vagueness, or partial relevance.
            - **4-3:** The response is largely incoherent, unclear, or mostly irrelevant.
            - **2-0:** The response is completely nonsensical, empty, or fails to address the question/task.

        ### Task Type: {task_type.capitalize()}
        - **Image ID**: {image_id}
        - **Question**: {question}
        - **Generated Answer**: {generated_answer}

        **Note**: This evaluation focuses only on the readability and meaningfulness of the generated text. 
        Do not consider factual accuracy. A well-written refusal should be scored highly.

        Please evaluate the meaningfulness of the generated response based on the rubric above, and return a score (0-10) along with a short justification.

        Example Output:
        {{
            "Meaningful Score": [Insert score from 0-10],
            "Justification": "[Optional] Provide a brief justification explaining why the meaningful score was assigned."
        }}
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "You are an expert at evaluating the meaningful of responses."
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                },
            ],
        )
        content_json = convert_markdown_json_to_json(completion.choices[0].message.content)
        if validate_structure(content_json)[0]:
            return content_json
        else:
            print(f"Invalid structure in response: {content_json}")
            return None
    except Exception as e:
        print(f"Error occurred while generating biography: {e}")
        return None


def process_single_question(question_data, output_data, generation_scores):
    question_type = question_data.get("question type")
    image_id = question_data.get("image_id")
    image_type = question_data.get("image_type")
    question = question_data.get("question")
    generated_answer = question_data.get("generated_answer")
    ground_truth = question_data.get("ground_truth")

    # Evaluate meaningful for generation questions
    evaluation = None
    for _ in range(MAX_RETRIES):
        evaluation = evaluate_meaningful_questions(image_id,
                                                   question,
                                                   generated_answer,
                                                   ground_truth,
                                                   task_type="generation")
        if evaluation is not None:
            break
    if evaluation is None:
        meaningful_score, justification = None, None
    else:
        meaningful_score, justification = evaluation["Meaningful Score"], evaluation["Justification"]

    with generation_lock:
        if meaningful_score is not None:
            generation_scores[question_type].append(meaningful_score)
        # Append the results to the output data
        output_data[question_type].append({
            "Task Type": "Generation",
            "Question Type": question_type,
            "Image_ID": image_id,
            "image_type": image_type,
            "Question": question,
            "Generated Answer": generated_answer,
            "Ground Truth": ground_truth,
            "Meaningful Score": meaningful_score,
            "Justification": justification
        })


def process_generation_questions(generation_questions, output_data, generation_scores):
    print("Processing generation questions with progress bar...")
    list(thread_map(lambda q: process_single_question(q, output_data, generation_scores), generation_questions))


def evaluate_meaningful_from_json(json_file_path, output_folder):
    # Read the input JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Prepare the output data list
    output_data = {"image_understanding": [], "image_memory": [], "text_memory": []}
    generation_scores = {"image_understanding": [], "image_memory": [], "text_memory": []}

    # Process Generation Questions
    generation_questions = data.get("Generation_Questions", [])
    process_generation_questions(generation_questions, output_data, generation_scores)

    # Calculate average scores
    avg_image_understanding_score = sum(generation_scores["image_understanding"]) / len(
        generation_scores["image_understanding"]) if generation_scores["image_understanding"] else 0
    avg_image_memory_score = sum(generation_scores["image_memory"]) / len(
        generation_scores["image_memory"]) if generation_scores["image_memory"] else 0
    avg_text_memory_score = sum(generation_scores["text_memory"]) / len(
        generation_scores["text_memory"]) if generation_scores["text_memory"] else 0

    # Append average scores to the output data
    output_data["Score"] = {
        "Average Image Understanding Score": avg_image_understanding_score,
        "Average Image Memory Score": avg_image_memory_score,
        "Average Text Memory Score": avg_text_memory_score,
    }

    # Define output filename with the new name
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_meaningful_score.json")

    # Save the results to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as output_f:
        json.dump(output_data, output_f, indent=4)

    print(f"Meaningful evaluation results saved to: {output_file}")


def count_evaluated_folders(input_folder, output_folder):
    total_folders = 0
    evaluated_folders = 0

    # Iterate through all subdirectories in the input folder
    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            total_folders += 1

            # Get all JSON files in the directory
            json_files = [f for f in os.listdir(subdir_path) if f.endswith(".json")]

            # Check if all the relevant files in this folder have been processed
            all_processed = True
            for filename in json_files:
                if filename.startswith(eval_set) and "_score" not in filename:
                    base_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(output_folder, f"{base_name}_meaningful_score.json")

                    if not os.path.exists(output_file):
                        all_processed = False
                        break

            # If all files are processed, count this folder as evaluated
            if all_processed:
                evaluated_folders += 1

    # Report the number of evaluated folders
    print(f"{evaluated_folders}/{total_folders} folders evaluated.")


def process_all_files_in_folder(input_folder, output_folder):
    # List all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file starts with the specified prefixes and ends with ".json"
        # Also, skip files that already contain "_score" to avoid reprocessing output files
        if (filename.startswith(eval_set) and filename.endswith(".json")
                and "_score" not in filename):  # Skip processed files

            json_file_path = os.path.join(input_folder, filename)

            # Check if the output file for this JSON file already exists
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, f"{base_name}_meaningful_score.json")

            if os.path.exists(output_file):
                print(f"Skipping {json_file_path}, already evaluated.")
                continue

            print(f"Processing file: {json_file_path}")
            evaluate_meaningful_from_json(json_file_path, output_folder)


def process_all_folders_in_eval_result(root_folder):
    # Iterate through all subdirectories in the root eval_result folder
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            print(f"Processing folder: {subdir_path}")
            process_all_files_in_folder(subdir_path, subdir_path)  # Use the same folder for input and output


def run_evaluation(input_folder):
    # Check if the folder contains JSON files directly
    contains_json_files = any(
        filename.startswith(eval_set) and filename.endswith(".json") for filename in os.listdir(input_folder))

    if contains_json_files:
        # If the folder contains JSON files directly, process the folder
        print(f"Processing a single folder: {input_folder}")
        process_all_files_in_folder(input_folder, input_folder)
    else:
        # If the folder does not contain JSON files directly, process all subdirectories
        print(f"Processing nested folders under: {input_folder}")
        process_all_folders_in_eval_result(input_folder)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")
    parser.add_argument('--input_folder', type=str, required=True, help='Model ID or path to the model.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_folder = args.input_folder
    count_evaluated_folders(input_folder, input_folder)
    run_evaluation(input_folder)

if __name__ == "__main__":
    main()
