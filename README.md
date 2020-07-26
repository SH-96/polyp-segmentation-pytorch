# polyp-segmentation-pytorch
This repository contains code to train a U-Net segmentation model to segment polyp images.

Data downloaded from: https://datasets.simula.no/hyper-kvasir/hyper-kvasir-segmented-images.zip

Download the data. Extract Stage 1 images and store with the following structure:
- data
  - images/
    - cju0qkwl35piu0993l0dewei2.jpg
    - cju0qoxqj9q6s0835b43399p4.jpg
  - masks/
    - cju0qkwl35piu0993l0dewei2.jpg
    - cju0qoxqj9q6s0835b43399p4.jpg
        
## Notebook Instructions:
1. Change default values in the notebook if needed.
2. Run the notebook.

## Manual Implementation Instructions:
1. Run utils.py with arguments:
  - --data_dir: Directory containing train images.
  - --text_dir: Directory to store train and test files.
  - --split_ratio: Dataset Split Size
  - --random_seed: Random seed for the split to ensure reproducibility
  
  Example: 
  ```python utils.py --data_dir data/ --text_dir data/ --split_ratio .80 .10 .10 --random_seed 7```

2. Make changes to the config file if needed.<br>
3. Run train.py with arguments:
  - --config: Path to config file
  
  Example: 
  ```python train.py --config config.yml```
  
4. Run test.py with arguments:
  - --output_dir: Output directory of the experiment
  - --threshold: Threshold for model prediction
  
  Example:
  ```python test.py --output_dir experiment_test/ --threshold 0.5```
