from CNN_model.CNN_attention import CNN_Attn
from CNN_model.constants import *
from torch.utils.data import DataLoader
from grid_models.grid_environment import GridEnvironment
import os
import torch
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
    # training_df = images_df.sample(n = 1000)



    testing_df = images_df.drop(index = training_df.index).sample(n = 10000)
    # testing_df = images_df.drop(index = training_df.index).sample(n = 1000)

    training_dataset = CustomImageLoadingDataset(training_df, train_val_transforms)
    training_loader = DataLoader(training_dataset, batch_size = 128, shuffle = True, num_workers = 1)

    testing_dataset = CustomImageLoadingDataset(testing_df, train_val_transforms)
    testing_loader = DataLoader(testing_dataset, batch_size = 128, shuffle = True, num_workers = 1)

    trial_nb = 45

    for pred_weights in tqdm([0.9]):

        saving_dirs = {
            'training_dir': f"../insights/Actor_Critic/Grid_models_training_trial{trial_nb}_W-{pred_weights}",
            'testing_dir': f"../insights/Actor_Critic/Grid_models_testing_trial{trial_nb}_W-{pred_weights}",
        }

    

        os.makedirs(saving_dirs['training_dir'], exist_ok=True)
        os.makedirs(f"{saving_dirs['training_dir']}/stats", exist_ok=True)
        os.makedirs(f"{saving_dirs['training_dir']}/images", exist_ok=True)
        os.makedirs(f"{saving_dirs['training_dir']}/models", exist_ok=True)

        os.makedirs(saving_dirs['testing_dir'], exist_ok=True)
        os.makedirs(f"{saving_dirs['testing_dir']}/stats", exist_ok=True)
        os.makedirs(f"{saving_dirs['testing_dir']}/images", exist_ok=True)
        os.makedirs(f"{saving_dirs['testing_dir']}/models", exist_ok=True)
        training_df.to_csv(f"{saving_dirs['training_dir']}/training_data.csv")
        testing_df.to_csv(f"{saving_dirs['testing_dir']}/testing_data.csv")

        w = 0.8
        total_episodes = 8

        target_episode = int(total_episodes * 1)
        sequenceRL = GridEnvironment(
            action_dim=1,
            saving_dirs=saving_dirs,
            training_loader=training_loader,
            testing_loader=testing_loader,
            ground_truth_model=model,
            target_episode= target_episode,
            grid_size= [12,12],
            nb_steps=50,
            threshold=0.4,
            weight_pred=pred_weights
        )
        
        sequenceRL.train_parallelized(total_episodes)
        sequenceRL.validate(sequenceRL.agent, total_episodes, sequenceRL.agent.rm)






