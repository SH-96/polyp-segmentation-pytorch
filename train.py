# Name: Sanjith Hebbar
# Date: 26-07-2020
# Description: Script to train U-net model for segmenting polyps

# Standard libararies
import os
import yaml
import argparse
from prettytable import PrettyTable

# Pytorch Libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from trainer_utils import *
from dataset_utils import PolypDataset
from models import UNet

# Ensure Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Path to config file for training.", required = True)
    args = parser.parse_args()

    # Read config file
    with open(args.config) as f:
        config_params = yaml.full_load(f)

    # Input size for model
    input_size = config_params['input_size']

    # Batch size for training
    batch_size = config_params['batch_size']

    # Create train and validation sets
    train_images_file = config_params["train_images_txt"]
    train_labels_file = config_params["train_masks_txt"]
    val_images_file = config_params["val_images_txt"]
    val_labels_file = config_params["val_masks_txt"]

    print("Loading Data...")
    # Initialise Datasets
    train_set = PolypDataset(train_images_file, train_labels_file, input_size)
    val_set = PolypDataset(val_images_file, val_labels_file, input_size)

    # Random seed for dataloader
    random_seed = config_params['seed']

    # Initialize Dataloaders
    torch.manual_seed(random_seed)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True)

    # Checkpoint path
    output_dir = config_params['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "polyp_unet.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialise Model
    print("Loading Model...")
    torch.manual_seed(random_seed)
    num_channels = config_params['num_channels']
    n_classes = config_params['n_classes']
    bilinear = config_params['bilinear']
    
    model = UNet(n_channels = num_channels, n_classes = n_classes, bilinear = bilinear).to(device)

    # Hyperparameters
    num_epochs = config_params['num_epochs']
    learning_rate = config_params['learning_rate']
    patience = config_params['patience']

    # Prettytable
    table = PrettyTable(field_names=['Parameter', "Value"])
    table.align = "l"
    table.add_row(["Output Directory", output_dir])
    table.add_row(["Model Path", save_path])
    table.add_row(["Epochs", num_epochs])
    table.add_row(["Patience", patience])
    table.add_row(["Learning Rate", learning_rate])
    table.add_row(["Device", device])
    table.add_row(["Seed", random_seed])
    table.add_row(["Input Size", input_size])
    table.add_row(["Batch Size", batch_size])
    print(table)

    print("Training Model...")
    # Train model
    best_model_params = train(model, train_loader, val_loader, batch_size, num_epochs, 
                              learning_rate, patience, save_path, device)

    config_params['model_checkpoint'] = save_path
    config_params['device'] = device.type
    new_config_path = os.path.join(output_dir, "config.yml")

    with open(new_config_path, 'w') as f:
        yaml.dump(config_params, f, default_flow_style = False)

    print("\nTraining Complete.\nModel stored at {0}".format(save_path))