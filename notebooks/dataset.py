import os
from PIL import Image
import numpy as np
from datasets import Dataset


# def get_data(conditioning_folder, real_folder, prompts):
#     """
#     Create a list of (conditioning image path, ground truth image path, prompt path) tuples from their path.

#     Args:
#         conditioning_folder (str): Path to conditioning images folder.
#         real_folder (str): Path to ground truth images folder.
#         prompts: list of string.

#     Returns:
#         tuple: Tuple containing conditioning images, ground truth images, and prompts.
#     """
        
#     dataset = []

#     conditioning_images = os.listdir(conditioning_folder)
#     real_images = os.listdir(real_folder)
   
    
#     for conditioning_image, real_image, prompt in zip(conditioning_images, real_images, prompts):
#         conditioning_image_path = os.path.join(conditioning_folder, conditioning_image)
#         real_image_path = os.path.join(real_folder, real_image)
        
#         # Open images and prompt
#         conditioning_image_data = Image.open(conditioning_image_path)
#         real_image_data = Image.open(real_image_path)
    
#         # Add to dataset dictionary
#         dataset.append((conditioning_image_data,real_image_data,prompt))
    
#     return dataset

# the input is now maybe different for later
def get_data(data_path):
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
    for image_folder in os.listdir(data_path):  # TODO only use a subset here cause dataset large
        image_path = os.path.join(data_path, image_folder)
        if os.path.isdir(image_path):
            outputs = [f for f in os.listdir(os.path.join(image_path, "color")) if f.endswith('.png')]
            targets = [f for f in os.listdir(os.path.join(image_path, "target")) if f.endswith('.png')]
            output_paths = [os.path.join(image_path, "color", output) for output in outputs]
            target_paths = [os.path.join(image_path, "target", target) for target in targets]
            prompts = ["The same image but fixing small physical and illumination inconsistencies"]*len(outputs)
            for o, t, p in zip(output_paths, target_paths, prompts):
                try:
                    dataset.append((Image.open(o), Image.open(t), p))
                except:  # Didn't happen but just to be sure
                    continue
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
    entries = get_data("re10k")
    dataset = create_dataset(entries)
    # dataset.push_to_hub("re10ksmalltest")  #dont know how large this can be
    # may need https://discuss.huggingface.co/t/uploading-files-larger-than-5gb-to-model-hub/4081
    # os.makedirs("/reallynicefolder")
    dataset.save_to_disk("/reallynicefolder")