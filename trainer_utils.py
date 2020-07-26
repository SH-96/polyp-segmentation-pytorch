# Name: Sanjith Hebbar
# Date: 26-07-2020
# Description: Script containing utility functions for training and testing

# Import Libraries
import torch
from torch import nn, optim
from models import DiceBCELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def train(model, train_dataloader, val_dataloader, batch_size, num_epochs, learning_rate, patience, model_path, device):
    """
    Function to train a u-net model for segmentation.
    model: U-Net model
    Args:
        train_dataloader: training set
        val_dataloader: validation set
        batch_size: batch_size for training.
        num_epochs: number of epochs to train.
        learning_rate: learning rate for the optimiser
        patience: number of epochs for early stopping.
        model_path: checkpoint path to store the model.
        device: CPU or GPU to train the model.
    
    Returns: A dictionary containing the training and validation losses.
    """
    
    # Loss Collection
    train_losses  = []
    val_losses = []

    # Loss function
    criterion = DiceBCELoss().to(device)

    # Optimiser
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min' if model.n_classes > 1 else 'max', patience=patience)

    # Patience count
    count = 0

    for epoch in tqdm(range(1, num_epochs+1)):
        current_train_loss = 0.0
        current_val_loss = 0.0

        # Train model
        model.train()
        for features, labels, idx in train_dataloader:
            optimiser.zero_grad()
            features, labels = features.to(device), labels.to(device)
            output = model.forward(features)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()
            current_train_loss += loss.item()

            del features, labels
            torch.cuda.empty_cache()
            
        # Evaluate model
        model.eval()
        with torch.no_grad():
            for features, labels, idx in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                output = model.forward(features)
                loss = criterion(output, labels)
                current_val_loss += loss.item()

                del features, labels
                torch.cuda.empty_cache()
        
        # Store Losses
        current_train_loss /= len(train_dataloader)
        train_losses.append(current_train_loss)

        current_val_loss /= len(val_dataloader)
        val_losses.append(current_val_loss)

        tqdm.write("Epoch: {0:d} -> Train Loss: {1:0.8f} Val Loss: {2:0.8f} ".format(epoch, current_train_loss, current_val_loss))
        if((epoch == 1) or (current_val_loss < best_val_loss)):
            best_val_loss = current_val_loss
            eq_train_loss = current_train_loss
            best_epoch = epoch
            count = 0

            # Save best model
            torch.save(model.state_dict(), model_path)
        
        # Check for patience level
        if(current_val_loss > best_val_loss):
            count += 1
            if(count == patience):
                break

    # Save best parameters
    best_model_params = {'train_losses': train_losses, 
                        'val_losses': val_losses, 
                        'best_val_loss': best_val_loss,
                        'eq_train_loss': eq_train_loss,
                        'best_epoch': best_epoch}
                             
    return best_model_params

def predict_mask(model, img, threshold, device):
    output = model.forward(img.to(device))
    output = torch.sigmoid(output).detach().cpu().numpy()
    pred = output > threshold

    return pred


def visualize(input_img, pred, mask):
    fig, axes = plt.subplots(1, 3, num = "U-Net Prediction", figsize=(15, 7), dpi=80, sharex=True, sharey=True)
    titles = ['Input', 'Prediction', 'GT Mask']
    image_sets = [input_img, pred, mask]
    for i, axis in enumerate(axes):
        if (i == 0):
            img = image_sets[i].squeeze(0).permute(1, 2, 0)
            axis.imshow(img)
        else:
            img = image_sets[i].squeeze()
            axis.imshow(img, cmap = 'gray')
        axis.set_title(titles[i])

    plt.show()