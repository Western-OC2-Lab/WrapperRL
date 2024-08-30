from CNN_model.CNN_attention import CNN_Attn
from CNN_model.constants import *
from torch.utils.data import DataLoader
from grid_models.grid_environment import GridEnvironment
import os
import torch
import pandas as pd
from Utils.image_loaders import CustomImageLoadingDataset
from tqdm import tqdm
from grid_models.sac_trainer import Sac_Trainer

if __name__ == '__main__':

    MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"

    model = CNN_Attn()
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    DATASET_PATH = f"{DATASET_DIR}/{DATASET_NAME}"

    images_df = pd.read_csv(DATASET_PATH, index_col = [0])

    training_df = images_df.sample(n = 1000)

    trial_nb = 45

    set_weights = [0.9]

    for w in set_weights:
        saving_dirs = {
            'training_dir': f"../insights/Actor_Critic/Grid_models_training_trial{trial_nb}_W-{w}",
            'testing_dir': f"../insights/Actor_Critic/Grid_models_testing_trial{trial_nb}_W-{w}",
        }   
        testing_df = pd.read_csv(f"{saving_dirs['testing_dir']}/testing_data.csv", index_col=[0])

        training_dataset = CustomImageLoadingDataset(training_df, train_val_transforms)
        training_loader = DataLoader(training_dataset, batch_size = 64, shuffle = True, num_workers = 1)

        testing_dataset = CustomImageLoadingDataset(testing_df, train_val_transforms)
        testing_loader = DataLoader(testing_dataset, batch_size = 64, shuffle = True, num_workers = 1)
        total_episodes = 8

        
        for run in range(10):
            saving_dirs = {
                'training_dir': f"../insights/Actor_Critic/Grid_models_training_trial{trial_nb}_W-{w}",
                'testing_dir': f"../insights/Actor_Critic/Grid_models_testing_trial{trial_nb}_W-{w}_R-{run}",
            } 

            os.makedirs(saving_dirs['testing_dir'], exist_ok=True)
            os.makedirs(f"{saving_dirs['testing_dir']}/stats", exist_ok=True)
            os.makedirs(f"{saving_dirs['testing_dir']}/images", exist_ok=True)
            

            target_episode = int(total_episodes * 0.8)
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
                weight_pred=w
            )

            for episode in tqdm(range(1, total_episodes+1)):
                agent = Sac_Trainer(
                    target_episode,
                    [72, 72],
                    [72, 72],
                    1,
                    24,
                    16,
                    1e-5,
                    sequenceRL,
                    saving_dirs,
                    1,
                    0.05,
                    0.15,
                    device = torch.device("cuda")
                )
                agent.load_models(episode)
                sequenceRL.validate(agent, episode, agent.rm)






