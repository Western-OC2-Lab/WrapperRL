import numpy as np
from grid_models.sac_models import *
import torch
import pandas as pd

# Calculate the percentage difference in weights
def calculate_weight_difference(net1, net2):
    net1_weights = np.array([])
    net2_weights = np.array([])

    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        net1_weights = np.concatenate((net1_weights, param1.detach().numpy().flatten()))
        net2_weights = np.concatenate((net2_weights, param2.detach().numpy().flatten()))

    weight_difference = np.abs(net1_weights - net2_weights)
    percentage_difference = (weight_difference / np.abs(net1_weights)) * 100
    average_difference = np.mean(percentage_difference)
    return average_difference

trial_nb = 45
W = 0.9

saving_dirs = {
        'training_dir': f"../insights/Actor_Critic/Grid_models_training_trial{trial_nb}_W-{W}",
        'testing_dir': f"../insights/Actor_Critic/Grid_models_testing_trial{trial_nb}_W-{W}",
    }


set_differences = {}
total_episodes = 8

for i in range(1, total_episodes):
    for j in range(i+1, total_episodes):
        name_res = f"X-{i}_Y-{j}"

        # pol_1 = TwinnedQNetwork(1375, 1, 37)

        # pol_2 = TwinnedQNetwork(1375, 1, 37)
        pol_1 = GaussianNNPolicy(1375, 1, 37)
        pol_2 = GaussianNNPolicy(1375, 1, 37)


        # pol_1.load_state_dict(torch.load(f"{saving_dirs['training_dir']}/models/EP-training_{i}_target_critic.pt"))
        # pol_2.load_state_dict(torch.load(f"{saving_dirs['training_dir']}/models/EP-training_{j}_target_critic.pt"))

        pol_1.load_state_dict(torch.load(f"{saving_dirs['training_dir']}/models/EP-training_{i}_actor.pt"))
        pol_2.load_state_dict(torch.load(f"{saving_dirs['training_dir']}/models/EP-training_{j}_actor.pt"))

        avg_diff = calculate_weight_difference(pol_1, pol_2)
        set_differences[name_res] = avg_diff

df_eq = pd.DataFrame.from_dict([set_differences])
df_eq.to_csv("actor_diff.csv")



