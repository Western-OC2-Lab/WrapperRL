from grid_models.random_patch import RandomPatch
import os
from CNN_model.CNN_attention import CNN_Attn
import torch
from torch.utils.data import DataLoader
from CNN_model.constants import *
import pandas as pd
from Utils.image_loaders import CustomImageLoadingDataset
from tqdm import tqdm

if __name__ == '__main__':

    MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"


    model = CNN_Attn()
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()
    DATASET_PATH = f"{DATASET_DIR}/{DATASET_NAME}"
    images_df = pd.read_csv(DATASET_PATH, index_col = [0])

    training_df = images_df.sample(n = 30000)
    training_dataset = CustomImageLoadingDataset(training_df, train_val_transforms)
    training_loader = DataLoader(training_dataset, batch_size = 128, shuffle = True, num_workers = 1)

    trial_nb =1
    saving_dirs = f"../insights/Actor_Critic/Grid_models_training_trial{trial_nb}_randomPatch"

    os.makedirs(saving_dirs, exist_ok=True)
    os.makedirs(f"{saving_dirs}/stats", exist_ok=True)
    os.makedirs(f"{saving_dirs}/images", exist_ok=True)
    os.makedirs(f"{saving_dirs}/models", exist_ok=True)
    training_df.to_csv(f"{saving_dirs}/training_data.csv")

    device = torch.device("cuda")

    rand_drop = RandomPatch(
        loaded_model=model,
        img_dim = 72,
        batch_size = 128,
        saving_dir=saving_dirs,
        device = device,
        grid_size=[12, 12],
        data_loader=training_loader,
        type_patch='random'
    )

    rand_drop.validate(10)