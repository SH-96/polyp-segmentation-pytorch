# Name: Sanjith Hebbar
# Date: 26-07-2020
# Description: Creating data files for training and testing u-net model.

# Import Libraries
import cv2
import os
import random
import argparse
import numpy as np

def split_data(data_dir, seed, split_ratio = [.80, .10, .10]):
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    
    data_dict = {}
    for image in os.listdir(img_dir):
        img_path = os.path.join(img_dir, image)
        mask_path = os.path.join(mask_dir, image)
        data_dict[image] = [img_path, mask_path]

    # Random Seed
    random.seed(seed)
    data_list = list(data_dict.items())
    random.shuffle(data_list)
    
        
    # Data split
    train_size = int(len(data_list)*float(split_ratio[0]))
    val_size = int(len(data_list)*float(split_ratio[1]))
    test_size = len(data_list) - train_size - val_size

    train_dict = dict(data_list[:train_size])
    val_dict = dict(data_list[train_size:train_size+val_size])
    test_dict = dict(data_list[train_size+val_size:])
    
    return train_dict, val_dict, test_dict

def create_text_file(data_dict, data_type, img_txt_path, mask_txt_path = None):
    """
    Function to create text files containing image paths and mask paths
    Args:
    data_dict: Dictionary containing image paths and mask paths
    data_type: "train" or "test" data
    img_text_path: Path of text file containing image paths.
    img_mask_path: Path of text file containing mask paths.

    Example:
    create_text_file(train_dict, "train", "train_images.txt", "train_masks.txt")
    """
    with open(img_txt_path, 'w') as f:
        with open(mask_txt_path, 'w') as i:
            for key, value in data_dict.items():
                f.write(value[0])
                i.write(value[1])

                f.write("\n")
                i.write("\n")
                
    print("\n\n{0} files created.\nImages: {1}\nMasks: {2}".format(data_type.capitalize(), img_txt_path, mask_txt_path))
    return

def get_image_sample(data_loader):
    # Get Sample
    inputs, masks, idx = next(iter(data_loader))
    
    # Display Masks
    fig, axes = plt.subplots(1, 2)
    titles = ['Input', 'Mask']
    image_sets = [inputs[0], masks[0]]
    for i, axis in enumerate(axes):
        if(i == 0):
            axis.imshow(image_sets[i].squeeze(0).permute(1, 2, 0))
        else:
            axis.imshow(image_sets[i].squeeze(), cmap = 'gray')
        axis.set_title(titles[i])

    print("Model Input Shape: ", inputs.shape)
    print("Masks Shape: ", masks.shape)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required = True, help = "Directory containing images and masks.")
    parser.add_argument('--text_dir', required = True, help = "Directory to store image and mask text files.")
    parser.add_argument('--split_ratio', required = True, nargs = "+", help = "Validation split size. Must be in decimals. Example: .80 .10 .10")
    parser.add_argument('--random_seed', required = True, help = "Random Seed for splitting validation set.")
    args = parser.parse_args()

    train_dict, val_dict, test_dict = split_data(args.data_dir, args.random_seed, args.split_ratio)

    train_img_txt = os.path.join(args.text_dir, "train_images.txt")
    train_mask_txt = os.path.join(args.text_dir, "train_masks.txt")
    val_img_txt = os.path.join(args.text_dir, "val_images.txt")
    val_mask_txt = os.path.join(args.text_dir, "val_masks.txt")
    test_img_txt = os.path.join(args.text_dir, "test_images.txt")
    test_mask_txt = os.path.join(args.text_dir, "test_masks.txt")

    create_text_file(train_dict, "train", train_img_txt, train_mask_txt)
    create_text_file(val_dict, "val", val_img_txt, val_mask_txt)
    create_text_file(test_dict, "test", test_img_txt, test_mask_txt)