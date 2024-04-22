import os
from PIL import Image
import numpy as np
from datasets import Dataset


def get_data(conditioning_folder, real_folder, prompts):
    """
    Create a list of (conditioning image path, ground truth image path, prompt path) tuples from their path.

    Args:
        conditioning_folder (str): Path to conditioning images folder.
        real_folder (str): Path to ground truth images folder.
        prompts: list of string.

    Returns:
        tuple: Tuple containing conditioning images, ground truth images, and prompts.
    """
        
    dataset = []

    conditioning_images = os.listdir(conditioning_folder)
    real_images = os.listdir(real_folder)
   
    
    for conditioning_image, real_image, prompt in zip(conditioning_images, real_images, prompts):
        conditioning_image_path = os.path.join(conditioning_folder, conditioning_image)
        real_image_path = os.path.join(real_folder, real_image)
        
        # Open images and prompt
        conditioning_image_data = Image.open(conditioning_image_path)
        real_image_data = Image.open(real_image_path)
    
        # Add to dataset dictionary
        dataset.append((conditioning_image_data,real_image_data,prompt))
    
    return dataset

def create_dataset(data):
    """
    Create a HuggingFace dataset from a set of (conditioning image, ground truth image, prompt) tuples.
    
    Args:
    - data (list): List of tuples containing (conditioning image, ground truth image, prompt).
    
    Returns:
    - Dataset: HuggingFace Dataset object.
    """
    dataset_dict = {
        "conditioning_image": [],
        "ground_truth_image": [],
        "prompt": []
    }
    
    for conditioning_image, ground_truth_image, prompt in data:
        # Convert images to NumPy arrays
        # conditioning_image = np.array(conditioning_image)
        # ground_truth_image = np.array(ground_truth_image)
        
        # Add data to dataset dictionary
        dataset_dict["conditioning_image"].append(conditioning_image)
        dataset_dict["ground_truth_image"].append(ground_truth_image)
        dataset_dict["prompt"].append(prompt)
        
    return Dataset.from_dict(dataset_dict)


if __name__ == "__main__":
    dataset = get_data(conditioning_folder, real_folder, prompts)
    create_dataset(dataset)