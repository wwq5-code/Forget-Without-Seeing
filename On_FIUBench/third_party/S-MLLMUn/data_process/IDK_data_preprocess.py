import pandas as pd
import json
from typing import Any
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import random

from .IDK import idk
def get_random_answer_from_idk():
    return random.choice(list(idk))

class IDK_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True, multimodal: bool = True, forget: bool=False):
        super().__init__()
        self.multimodal = multimodal
        self.forget = forget
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        flattened_data = []

        for idx, row in self.df.iterrows():
            character_image_data = row['character_image'].get('bytes')  # Access the image bytes
            ophthalmic_image_data = row['ophthalmic_image'].get('bytes')
            # Convert the image bytes to a PIL Image
            try:
                character_image = Image.open(BytesIO(character_image_data)).convert("RGB")
                ophthalmic_image = Image.open(BytesIO(ophthalmic_image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            # Safely load metadata as JSON
            try:
                metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue

            for qa_pair in metadata:
                type = qa_pair.get("Type", "")
                question = qa_pair.get("Question", "")
                if self.forget:
                    answer = get_random_answer_from_idk()
                else:
                    answer = qa_pair.get("Answer", "")
                if type and question and answer:
                    if self.multimodal:
                        if type == "multimodal":
                            flattened_data.append({"image": character_image, "question": question, "answer": answer})
                        elif type == "ophthalmic":
                            flattened_data.append({"image": ophthalmic_image, "question": question, "answer": answer})
                    else:
                        if type == "pure_text":
                            flattened_data.append({"image": None, "question": question, "answer": answer})
        return flattened_data

    def resize_image(self, image):
        """
        Resizes the image to half of its original dimensions if target_size is not specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image to half the original size if target_size is None.
        """
        if image is None:
            return None
        if self.target_size is None:
            # Calculate half of the original image size
            half_width = image.width // 2
            half_height = image.height // 2
            return image.resize((half_width, half_height), Image.Resampling.LANCZOS)
        else:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {"image": image, "question": tokenized_question, "answer": tokenized_answer}

def train_collate_fn(examples, processor, args):
    images = []
    texts = []
    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')
        if image:
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "image"
                }, {
                    "type": "text",
                    "text": question
                }]
            }, {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": answer
                }]
            }]
            images.append(image)
        else:
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": question
                }]
            }, {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": answer
                }]
            }]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        texts.append(prompt)

    if len(texts) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")
    if len(images) == 0:
        batch = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
    else:
        batch = processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")
    # Process the batch
    

    # Mask labels: only compute loss on assistant's answers
    labels = batch["input_ids"].clone()
    user_prompt_end_token = processor.tokenizer.convert_tokens_to_ids("assistant")  # Suppose there is a special marker used to indicate the start of answer generation.

    for i in range(labels.shape[0]):
        user_prompt_end_pos = (labels[i] == user_prompt_end_token).nonzero(as_tuple=True)[0]
        if len(user_prompt_end_pos) > 0:
            labels[i, :user_prompt_end_pos.item() + 2] = -100  # Mask user input part

    batch["labels"] = labels
    return batch


