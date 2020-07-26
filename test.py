# Name: Sanjith Hebbar
# Date: 26-07-2020
# Description: Script to test U-net model for segmenting polyps

# Standard Libraries
import argparse
import os
import yaml

# Pytorch Libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# Custom Modules
from dataset_utils import PolypDataset
from models import UNet
from trainer_utils import predict_mask, visualize

# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help = "Path to model output directory.", required = True)
    parser.add_argument("--threshold", help = "Threshold for prediction.", required = True)
    args = parser.parse_args()

    # Read config file
    config_file = os.path.join(args.output_dir, "config.yml")
    with open(config_file) as f:
        config_params = yaml.full_load(f)

    # Get test set
    model_path = config_params['model_checkpoint']
    test_images_txt = config_params['test_images_txt']
    test_masks_txt = config_params['test_masks_txt']
    input_size = config_params['input_size']
    num_channels = config_params['num_channels']
    n_classes = config_params['n_classes']
    bilinear = config_params['bilinear']

    # torch.manual_seed(config_params['seed'])
    test_set = PolypDataset(test_images_txt, test_masks_txt, input_size)
    test_loader = DataLoader(test_set, batch_size = 1, sampler = RandomSampler(test_set))

    # Inference device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Model
    model = UNet(n_channels = num_channels, n_classes = n_classes, bilinear = bilinear).to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()

    # Threshold for prediction
    threshold = float(args.threshold)

    # Get test image
    img, mask, idx = next(iter(test_loader))
    pred = predict_mask(model, img, threshold, device)

    # Visualise Prediction
    visualize(img, pred, mask)