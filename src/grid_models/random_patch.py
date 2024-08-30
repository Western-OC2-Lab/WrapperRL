from Utils.actor_critic_utils import *
from CNN_model.constants import reverse_image_transform
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import pandas as pd

class RandomPatch:

    def __init__(self, loaded_model, img_dim, batch_size, saving_dir, 
                 device, grid_size, data_loader, type_patch):
        
        self.device = device
        self.loaded_model = loaded_model.to(self.device)
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.saving_dir = saving_dir
        self.grid_size = grid_size
        self.data_loader = data_loader
        self.type_patch = type_patch
        self.saving_dir = saving_dir

        self.utils = Utils([self.img_dim, self.img_dim])
        self.nb_cols, self.nb_rows = (self.img_dim // grid_size[0]), (self.img_dim // grid_size[1])
        self.len_indices = self.nb_cols * self.nb_rows
        self.action_dim = self.len_indices
        self.action_dim = 1

        self.curr_env = {
            'attention_map': None,
            'state': None,
            'reward': 0,
            'nb_steps': 0,
            'set_actions': [],
            'set_rewards': [],
            'img': None,
            'label': None,
            'name_img': None,
            'model_pred': None,
            'curr_step': 0,
            'agg_pred': [],
            'curr_dims': 0,
            'uniqueness_counter': 0
        }

    def map_actions_to_dims(self):
        set_curr_actions = []
        set_indices = []
        set_indices_arr = np.zeros(shape = (self.len_indices))
        set_dims = 0
        for action in self.curr_env['set_actions']:
            indx, dimensions = self.one_dimension_action_state(action[0])
            set_curr_actions.append(dimensions)
            set_indices.append(indx)
            set_indices_arr[indx] = 1
            X_start, grid_width, Y_start, grid_height = dimensions[0], dimensions[1], dimensions[2], dimensions[3]
            set_dims += (grid_height * grid_width)
        unique_indices = np.unique(set_indices, return_index=True)
        set_indices = np.array(set_indices)
        set_curr_actions = np.array(set_curr_actions)
        set_indices = set_indices[unique_indices[1]]
        set_curr_actions = set_curr_actions[unique_indices[1]]

        self.curr_env['curr_dims'] = (set_dims) / (self.img_dim * self.img_dim) - ((self.grid_size[0] * self.grid_size[1]) / (self.img_dim * self.img_dim))

        return set_curr_actions, set_indices, set_indices_arr
    
    def one_dimension_action_state(self, grid_position):
        row_idx, col_idx = grid_position // (self.nb_rows), grid_position % (self.nb_cols)

        X_start, Y_start = (col_idx * self.grid_size[0]), (row_idx * self.grid_size[0])
        
        return grid_position, [X_start, self.grid_size[0], Y_start, self.grid_size[0]]
    
    def transform_image(self, tensor_array):
        tensor_array = torch.mul(tensor_array, 255).numpy()
        instance_arr = (tensor_array + 1) /2
        instance_arr = tensor_array
        instance_arr[instance_arr > 1.0] = 1.0
        instance_arr[instance_arr < 0.0] = 0.0
        instance_arr = np.array(instance_arr, np.float32) * 255
        instance_arr = instance_arr.astype(np.uint8)

        return instance_arr
    
    def plot_predictions(self, saving_dir, r_weights, pred_test, orig_img, name_img, label, ep, batch):
        name_img = name_img.split("/")[-1]
        fig, ax = plt.subplots(1, 2, figsize = (15, 15))
        axes = ax.flatten()
        plotted_values_new = self.transform_image(r_weights[0, :, :, :].permute(1,2, 0))
        plotted_values_img = self.transform_image(orig_img[0, :, :, :].permute(1, 2, 0))
        
        axes[0].imshow(plotted_values_new)
        axes[1].imshow(plotted_values_img)

        avg_reward  = np.mean(self.curr_env['set_rewards'])
        plt.suptitle(f"Pred: {np.round(pred_test.item(), 3)}, Label: {label.item()}, Nb_steps: {len(self.curr_env['set_rewards'])}, Reward: {avg_reward}", fontsize = 35)

        plt.tight_layout()
        plt.savefig(f"{saving_dir}/{name_img}_{ep}_{batch}.png", pad_inches = 0, bbox_inches = 'tight')
        plt.close()


    def aggregate_map_acc(self, saving_dir, ep, batch_idx, type_val = 'train'):
        set_curr_actions, _, _ = self.map_actions_to_dims()

        # r_weights, _ = self.utils.deselect_region(self.curr_env['img'], set_curr_actions)
        r_weights, _ = self.utils.select_region(self.curr_env['img'], set_curr_actions)

        pred_test, _, _, _ = self.loaded_model(r_weights.cuda())
        self.curr_env['agg_pred'] = pred_test.item()
        
        r_weights, orig_img = self.utils.select_region(reverse_image_transform(self.curr_env['img']),set_curr_actions)
        
        saving_dir = f"{saving_dir}/images/ep_{ep}"
        os.makedirs(saving_dir, exist_ok=True)

        freq = 500 if type_val == 'train' else 25
        # freq = 2 if type_val == 'train' else 1

        if batch_idx % freq == 0:

            self.plot_predictions(saving_dir, r_weights, pred_test, orig_img, self.curr_env['name_img'], self.curr_env['label'], ep, batch_idx)
        
        return pred_test
    
    def check_actions_exists(self, action):
        curr_idx, _ = self.one_dimension_action_state(action[0])

        for p_action in self.curr_env['set_actions'][:-1]:
            indx, _ = self.one_dimension_action_state(p_action[0])
            if indx == curr_idx:
                return True
            
        else:
            return False
    
    def patch_random(self):
        random_patch = np.random.randint(0, self.len_indices)
        self.curr_env['set_actions'].append(random_patch)
        idx, _ = self.one_dimension_action_state(random_patch)
        done = False

        set_curr_actions, _, set_indices = self.map_actions_to_dims()
        r_weights, _ = self.utils.select_region(self.curr_env['img'], set_curr_actions)
        with torch.no_grad():
            pred_test, _, _, _ = self.loaded_model(r_weights.cuda())

        pred_reward = (self.curr_env['label']) * (pred_test[0]) + (1-self.curr_env['label']) * (1-pred_test[0])
        orig_reward = (self.curr_env['label']) * (self.curr_env['model_pred']) + (1-self.curr_env['label']) * (1-self.curr_env['model_pred'])
        reward = (self.weight_pred)* pred_reward +(1-self.weight_pred) *(1- self.curr_env['curr_dims'] * self.curr_env['curr_dims'])


        if orig_reward - pred_reward <= self.perf_deg:
            done, self.deg_violation = True, True
             

        if self.check_actions_exists(random_patch):
            self.curr_env['uniqueness_counter'] += 1
            reward = torch.tensor(-0.2)
        
        if self.curr_env['curr_step'] >= self.nb_steps or self.curr_env['uniqueness_counter'] >= self.uniqueness_counter:
            done, self.steps_violation = True, True

        self.curr_env['set_rewards'].append(reward.cpu().item())

        return random_patch, reward, done, pred_test, set_indices
    
    def reset_episode(self):
        self.curr_env = {
            'set_actions': [],
            'set_rewards': [],
            'img': None,
            'label': None,
            'name_img': None,
            'model_pred': None,
            'curr_step': 0,
            'agg_pred': [],
            'curr_dims': 0,
            'uniqueness_counter': 0
        }

        self.deg_violation = False
        self.above_thresh = False
        self.steps_violation = False


    def single_batch_validation(self, set_imgs, set_labels, set_names, batch_idx, episode, replay_memory):
        detailed_results, overview_results = [], []
        for index, img in enumerate(set_imgs):
            self.curr_env['img'] = img[None, :]
            self.curr_env['label'] = set_labels[index]
            self.curr_env['name_img'] = set_names[index]
            with torch.no_grad():
                act_pred, _, feature_map, _ = self.loaded_model(self.curr_env['img'].cuda())
                self.curr_env['model_pred'] = act_pred
                _, w, h = feature_map.size()

            avg_reward = 0
            while(self.above_thresh == False and self.deg_violation == False and self.steps_violation == False):

                label_value = self.curr_env['label'].float().reshape(1).reshape(-1,1)
                _, _, set_indices = self.map_actions_to_dims()
                # set_indices = self.r_all_action_indices(curr_actions)
                # action, log_value, mean_value, pred_agent, critic_val, pred_reward = agent_model.validate(self.curr_env['state'], set_indices, label_value, replay_memory)
                action, pred_reward, done, pred_agent, set_indices = self.patch_random()
                # idx, _ = self.one_dimension_action_state(action)
                avg_reward += pred_reward.item()
                # r = [action, log_value[0].item(), mean_value[0].item(), idx, pred_agent.item(), critic_val.item(), self.curr_env['curr_step'], self.curr_env['name_img']]
                r = [action, pred_agent.item(), self.curr_env['curr_step'], self.curr_env['name_img']]

                detailed_results.append(r)
                self.curr_env['curr_step'] += 1

            self.aggregate_map_acc(self.saving_dir, episode, batch_idx, 'testing')

            some_results = [self.curr_env['name_img'], self.curr_env['label'].item(), self.curr_env['model_pred'].item(), self.curr_env['agg_pred'], self.curr_env['curr_step'], self.curr_env['curr_dims'], (avg_reward)]
            
            overview_results.append(some_results)
            self.reset_episode()

        return (detailed_results, overview_results) 
    

    def validate(self, episode):
        
        start_time = time.time()

        all_results = Parallel(n_jobs=int(12), verbose=0)(delayed(self.single_batch_validation)(set_imgs, set_labels, set_names, batch_idx, episode) for batch_idx, (set_imgs, set_labels, set_names) in tqdm(enumerate(self.data_loader), total=len(self.data_loader)))

        end_time = time.time()

        validation_time = round((end_time - start_time) / 60, 4)
        print('validation_time:', validation_time)
        df_details_arr, df_m_details_arr = [], []
        for arr in all_results:
            df_details_arr.extend(arr[0])
            df_m_details_arr.extend(arr[1])

        df_details = pd.DataFrame(df_details_arr, columns = ["action", 'pred_agent', 'curr_step', 'name_img'])
        df_details.to_csv(f"{self.saving_dir}/stats/res_validation_{episode}.csv")

        df_m_details = pd.DataFrame(df_m_details_arr, columns = ['name_img', 'label', 'model_pred', 'agg_pred', 'curr_step', 'curr_dims', 'avg_reward'])
        df_m_details.to_csv(f"{self.saving_dir}/stats/res_validation_{episode}_details.csv")

        self.reset_episode()