from CNN_model.CNN_attention import CNN_Attn
from CNN_model.constants import *
from torch.utils.data import DataLoader
# from Intr_models.actor_critic_intr_models import IntersectionModels
from grid_models.grid_environment import GridEnvironment
import os
from grid_models.patch_drop import PatchDrop

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

    # training_df = images_df.sample(n = 30000)
    training_df = images_df.sample(n = 1000)

    trial_nb = 43

    saving_dirs = {
            'training_dir': f"../insights/Actor_Critic/Grid_models_training_trial{trial_nb}_patchDrop",
            'testing_dir': f"../insights/Actor_Critic/Grid_models_testing_trial{trial_nb}_patchDrop",
        }  
    # testing_df = images_df.drop(index = training_df.index).sample(n = 1000)
    testing_df = pd.read_csv(f"{saving_dirs['testing_dir']}/testing_data.csv", index_col=[0])
    # testing_df = images_df.sample(n =50)

    # testing_df = images_df.drop(index = training_df.index)
    # testing_df = testing_df.sample(n = 1000)
    training_dataset = CustomImageLoadingDataset(training_df, train_val_transforms)
    training_loader = DataLoader(training_dataset, batch_size = 64, shuffle = True, num_workers = 1)

    testing_dataset = CustomImageLoadingDataset(testing_df, train_val_transforms)
    testing_loader = DataLoader(testing_dataset, batch_size = 64, shuffle = True, num_workers = 1)

    

    os.makedirs(saving_dirs['training_dir'], exist_ok=True)
    os.makedirs(f"{saving_dirs['training_dir']}/stats", exist_ok=True)
    os.makedirs(f"{saving_dirs['training_dir']}/images", exist_ok=True)
    os.makedirs(f"{saving_dirs['training_dir']}/models", exist_ok=True)

    os.makedirs(saving_dirs['testing_dir'], exist_ok=True)
    os.makedirs(f"{saving_dirs['testing_dir']}/stats", exist_ok=True)
    os.makedirs(f"{saving_dirs['testing_dir']}/images", exist_ok=True)
    os.makedirs(f"{saving_dirs['testing_dir']}/models", exist_ok=True)
    w = 0.8
    total_episodes = 10

    target_episode = int(total_episodes * 0.8)
    # sequenceRL = IntersectionModels(saving_dirs, w, training_loader, testing_loader, model, target_episode)
    # sequenceRL = GridEnvironment(
    #     action_dim=1,
    #     saving_dirs=saving_dirs,
    #     training_loader=training_loader,
    #     testing_loader=testing_loader,
    #     ground_truth_model=model,
    #     target_episode= target_episode,
    #     grid_size= [12,12],
    #     nb_steps=50,
    #     threshold=0.4,
    #     weight_pred=0.5
    # )

    for episode in tqdm(range(0, total_episodes)):
        device = torch.device("cuda")
        agent = PatchDrop(loaded_model= model, target_episode = 10, img_dim = 72, state_dim = 1375, lr = 3e-4, batch_size = 64, saving_dir = saving_dirs,
                 alpha = 0.8, device = device, grid_size = [12, 12], training_loader = training_loader, validation_loader=testing_loader)
        agent.load_models(episode)
        # sequenceRL.train(episode)
        agent.validate(agent, episode)






