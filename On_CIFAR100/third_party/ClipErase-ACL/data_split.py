import pandas as pd
import os
import shutil

def split_dataset_by_keyword(csv_path, image_folder, keyword='elephant'):
    """
    Split a dataset into Forget Set and Retain Set based on a keyword in captions.

    Args:
        csv_path (str): Path to the CSV file containing image_name and comment columns.
        image_folder (str): Directory containing the image files.
        keyword (str): Keyword to determine which samples go to the Forget Set.
    """
    try:
        df = pd.read_csv(csv_path, sep='|', engine='python')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Strip extra whitespace in column names
    df.columns = [col.strip() for col in df.columns]

    # Ensure required columns exist
    if 'comment' not in df.columns or 'image_name' not in df.columns:
        print("Missing required columns 'comment' or 'image_name' in CSV file.")
        return

    # Output folders (you may change this to your own path)
    forget_set_folder = '/home/tianyu/dataset/flickr30k_images/forget_set'  # <-- change to your own path
    retain_set_folder = '/home/tianyu/dataset/flickr30k_images/retain_set'  # <-- change to your own path

    os.makedirs(forget_set_folder, exist_ok=True)
    os.makedirs(retain_set_folder, exist_ok=True)

    # Identify images with captions containing the keyword (case-insensitive)
    keyword_images = df[df['comment'].str.contains(keyword, case=False, na=False)]['image_name'].unique()

    # Forget Set: all samples for images containing the keyword
    forget_set = df[df['image_name'].isin(keyword_images)]

    # Retain Set: all other samples
    retain_set = df[~df['image_name'].isin(keyword_images)]

    print(f"Copying Forget Set images to {forget_set_folder} ...")
    for image in forget_set['image_name'].unique():
        try:
            shutil.copy(os.path.join(image_folder, image), forget_set_folder)
        except Exception as e:
            print(f"Error copying image {image}: {e}")

    print(f"Copying Retain Set images to {retain_set_folder} ...")
    for image in retain_set['image_name'].unique():
        try:
            shutil.copy(os.path.join(image_folder, image), retain_set_folder)
        except Exception as e:
            print(f"Error copying image {image}: {e}")

    # Save captions to CSV
    forget_set.to_csv(os.path.join(forget_set_folder, 'forget_set_captions.csv'), index=False)
    retain_set.to_csv(os.path.join(retain_set_folder, 'retain_set_captions.csv'), index=False)

    print(f"\n✅ Forget Set contains {len(forget_set['image_name'].unique())} images with captions.")
    print(f"✅ Retain Set contains {len(retain_set['image_name'].unique())} images with captions.")
    print("📁 Dataset split complete.")


if __name__ == "__main__":
    # Update the following paths to match your local environment
    csv_path = '/home/tianyu/dataset/flickr30k_images/results.csv'       # <-- change to your own path
    image_folder = '/home/tianyu/dataset/flickr30k_images'               # <-- change to your own path
    split_dataset_by_keyword(csv_path, image_folder, keyword='elephant')
