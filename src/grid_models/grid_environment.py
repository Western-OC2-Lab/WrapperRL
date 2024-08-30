from Utils.actor_critic_utils import *
from CNN_model.constants import reverse_image_transform
# from grid_models.actor_critic_trainer import Actor_Trainer
from grid_models.sac_trainer import Sac_Trainer
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, cpu_count, delayed
import time

class GridEnvironment:

    def __init__(self, action_dim, saving_dirs, training_loader, testing_loader, ground_truth_model, target_episode, grid_size, weight_pred, img_dim = 72, threshold = 0.3, nb_steps = 10,
                 perf_deg = 0.5, lr = 5e-4, discount_factor = 1.0, temperature = 0.25, grad_clip = 0):
        self.action_dim = action_dim # The step function changes based on the action dim
        self.saving_dirs = saving_dirs
        self.grid_size = grid_size
        self.img_dim = img_dim # Defines the image's dimensions
        self.nb_cols, self.nb_rows = (self.img_dim // grid_size[0]), (self.img_dim // grid_size[1])
        self.len_indices = self.nb_cols * self.nb_rows + 1
        self.weight_pred = weight_pred

        # Self explanatory
        self.training_loader = training_loader
        self.testing_loader = testing_loader

        self.utils = Utils([self.img_dim,self.img_dim])
        self.curr_step = 0
        self.deg_violation = False
        self.above_thresh = False
        self.steps_violation = False
        self.uniqueness_counter = 10
        self.temperature = temperature
        self.device = torch.device("cuda")

        # This variable stores the model responsible for classification results
        self.loaded_model = ground_truth_model
        self.loaded_model = self.loaded_model.to(self.device)
        self.loaded_model.eval()

        self.target_episode = target_episode # Defines when the exploration ends
        self.threshold = threshold # Defines the maximum dimensions of the de-selected area
        self.nb_steps = nb_steps # Defines the maximum number of steps in a single episode. Used jointly with the threshold to prevent selecting the same action multiple time
        self.perf_deg = perf_deg # Defines the maximum tolerable performance degradation compared to the original classification results

        self.batch_size = 64
        self.size_buffer = 200000



        self.discount_factor = discount_factor
        self.lr = lr
        self.model_pred = []
        
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

        self.saved_information = {
            'model_pred': [],
            'labels': [],
            'agg_pred': [],
            'img_name': []
        }

        agent = Sac_Trainer(
            self.target_episode,
            self.img_dim,
            self.img_dim,
            self.action_dim,
            self.size_buffer,
            self.batch_size,
            self.lr,
            self,
            self.saving_dirs,
            self.discount_factor,
            self.temperature,
            grad_clip,
            device = self.device
        )

        self.agent = agent
    
    def reset_episode(self):
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

        self.deg_violation = False
        self.above_thresh = False
        self.steps_violation = False


    # action = [index]
    def one_dimension_action_state(self, action):
        normalized_value = (action + 1) / 2
        grid_position = int(normalized_value * (self.len_indices-1))
        

        if grid_position < (self.len_indices-1):

            row_idx, col_idx = grid_position // (self.nb_rows), grid_position % (self.nb_cols)

            X_start, Y_start = (col_idx * self.grid_size[0]), (row_idx * self.grid_size[0])
            return grid_position, [X_start, self.grid_size[0], Y_start, self.grid_size[0]]
    
        else:
            return grid_position, [0, 0, 0, 0]


    def define_state_space(self, tensor_image: torch.Tensor):
        # new_tensor_image, dimensions = tensor_image.clone(), None
        new_tensor_image, dimensions = torch.zeros_like(tensor_image), None
        new_img  = tensor_image.clone()

        set_dims = 0
        set_indices = []
        for action in self.curr_env['set_actions']:
            if self.action_dim == 1:
                indx, dimensions = self.one_dimension_action_state(action[0])
            elif self.action_dim == 2:
                indx, dimensions = self.two_dimension_action_state(action[0])
            else:
                indx, dimensions = self.three_dimension_action_state(action[0])
            X_start, grid_width, Y_start, grid_height = dimensions[0], dimensions[1], dimensions[2], dimensions[3]
            # print(dimensions)
            new_tensor_image[:, :, int(Y_start):int(Y_start+grid_height), int(X_start):int(X_start+grid_width)] = new_img[:, :, int(Y_start):int(Y_start+grid_height), int(X_start):int(X_start+grid_width)]
            if indx < self.len_indices:
                set_dims += (grid_height * grid_width)
            set_indices.append(indx)

        unique_indices = np.unique(np.array(set_indices), return_index = True)
        diff_values = len(set_indices) - len(unique_indices[0])

        self.curr_env['curr_dims'] = (set_dims) / (self.img_dim * self.img_dim) - ((diff_values * self.grid_size[0] * self.grid_size[1]) / (self.img_dim * self.img_dim))
        return new_tensor_image
            
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

        # all_rewards = self.curr_env['set_predictions']
        # all_rewards = self.curr_env['set_rewards']
        # all_rewards = [f"EP-{idx}_{np.round(r.item(), 2)}" for idx, r in enumerate(all_rewards)]
        # string_reward = "_".join(all_rewards)
        # last_reward = np.round(self.curr_env['set_rewards'][-1],2)[0]
        avg_reward  = np.mean(self.curr_env['set_rewards'])
        # avg_reward = np.round(np.mean(self.curr_env['set_rewards']), 2)
        # plt.suptitle(f"Pred: {np.round(pred_test.item(), 3)}, Label: {label.item()}, Reward: {string_reward}", fontsize = 35)
        plt.suptitle(f"Pred: {np.round(pred_test.item(), 3)}, Label: {label.item()}, Nb_steps: {len(self.curr_env['set_rewards'])}, Reward: {avg_reward}", fontsize = 35)

        plt.tight_layout()
        plt.savefig(f"{saving_dir}/{name_img}_{ep}_{batch}.png", pad_inches = 0, bbox_inches = 'tight')
        plt.close()

    def map_actions_to_dims(self):
        set_curr_actions = []
        set_indices = []
        set_indices_arr = np.zeros(shape = (self.len_indices))
        for action in self.curr_env['set_actions']:
            if self.action_dim == 1:
                indx, dimensions = self.one_dimension_action_state(action[0])
            elif self.action_dim == 2:
                indx, dimensions = self.two_dimension_action_state(action[0])
            else:
                indx, dimensions = self.three_dimension_action_state(action[0])
            set_curr_actions.append(dimensions)
            set_indices.append(indx)
            set_indices_arr[indx] = 1
        unique_indices = np.unique(set_indices, return_index=True)
        set_indices = np.array(set_indices)
        set_curr_actions = np.array(set_curr_actions)
        set_indices = set_indices[unique_indices[1]]
        set_curr_actions = set_curr_actions[unique_indices[1]]
        return set_curr_actions, set_indices, set_indices_arr
    
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
    
    def parallelized_images(self, img, label, name, episode, batch_idx):
        self.curr_env['img'] = img[None, :]
        self.curr_env['label'] = label
        self.curr_env['name_img'] = name

        with torch.no_grad():
            act_pred, _, feature_map, _ = self.loaded_model(self.curr_env['img'])
            self.curr_env['model_pred'] = act_pred
            _, w, h = feature_map.size()
            self.curr_env['attention_map'] = F.interpolate(feature_map.view(1, 1, w, h), (self.img_dim, self.img_dim), mode = 'bilinear').view(1, 72, 72)


        while(self.above_thresh == False and self.deg_violation == False and self.steps_violation == False):
            # self.curr_env['state'] = self.define_state_space(self.curr_env['attention_map'])
            self.curr_env['state'] = self.define_state_space(self.curr_env['img'])

            label_value = self.curr_env['label'].float().reshape(1).reshape(-1,1)
            self.agent.train(self.curr_env['state'], label_value, episode)
            self.curr_env['curr_step'] += 1


        self.aggregate_map_acc(self.saving_dirs['training_dir'], episode, batch_idx, 'train')
        self.saved_information['labels'].append(self.curr_env['label'].item())
        self.saved_information['model_pred'].append(self.curr_env['model_pred'].item())
        self.saved_information['agg_pred'].append(self.curr_env['agg_pred'])
        self.saved_information['img_name'].append(self.curr_env['name_img'])
        self.reset_episode()


    def nn_state_space(self):

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output[0]
            return hook
        self.loaded_model.attn2.register_forward_hook(get_activation('attn2'))

        set_curr_actions, _, _ = self.map_actions_to_dims()
        ll = self.define_state_space(self.curr_env['img'])
        # r_weights, _ = self.utils.deselect_region(self.curr_env['img'], set_curr_actions)
        r_weights, _ = self.utils.select_region(self.curr_env['img'], set_curr_actions)

        
        _, _, attn_part, _ = self.loaded_model(r_weights.cuda())

        fc_image = activation['attn2']
        B, C, H, W = fc_image.size()
        feature_part = fc_image.reshape(B, C*H*W)
        B, H, W = attn_part.size()

        attn_part = attn_part.reshape(B, H*W)

        final_state = torch.cat((feature_part, attn_part), dim = 1)

        return final_state

    def single_batch_training(self, set_imgs, set_labels, set_names, batch_idx, episode):
        set_transitions = []
        start_batch_time = time.time()
        for index, img in enumerate(set_imgs):
            self.curr_env['img'] = img[None, :]
            self.curr_env['label'] = set_labels[index]
            self.curr_env['name_img'] = set_names[index]
            with torch.no_grad():
                act_pred, _, feature_map, _ = self.loaded_model(self.curr_env['img'].cuda())
                self.curr_env['model_pred'] = act_pred
                _, w, h = feature_map.size()
                self.curr_env['attention_map'] = F.interpolate(feature_map.view(1, 1, w, h), (self.img_dim, self.img_dim), mode = 'bilinear').view(1, 72, 72)

            while(self.above_thresh == False and self.deg_violation == False and self.steps_violation == False):
                self.curr_env['state'] = self.nn_state_space()
                label_value = self.curr_env['label'].float().reshape(1).reshape(-1,1)
                _, _, set_indices = self.map_actions_to_dims()
                curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step, set_indices = self.agent.train(self.curr_env['state'], label_value, set_indices, episode)
                self.curr_env['curr_step'] += 1
                transition = ( curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step, set_indices)
                set_transitions.append(transition)

            self.aggregate_map_acc(self.saving_dirs['training_dir'], episode, batch_idx, 'train')
            self.saved_information['labels'].append(self.curr_env['label'].item())
            self.saved_information['model_pred'].append(self.curr_env['model_pred'].item())
            self.saved_information['agg_pred'].append(self.curr_env['agg_pred'])
            self.saved_information['img_name'].append(self.curr_env['name_img'])
            self.reset_episode()
            

        end_batch_time = time.time()

        single_batch_running_time = round((end_batch_time - start_batch_time)/60, 4)

        # print('Single batch running time', single_batch_running_time)

        return set_transitions
    
    def single_batch_validation(self, agent_model, set_imgs, set_labels, set_names, batch_idx, episode, replay_memory):
        detailed_results, overview_results = [], []
        for index, img in enumerate(set_imgs):
            self.curr_env['img'] = img[None, :]
            self.curr_env['label'] = set_labels[index]
            self.curr_env['name_img'] = set_names[index]
            with torch.no_grad():
                act_pred, _, feature_map, _ = self.loaded_model(self.curr_env['img'].cuda())
                self.curr_env['model_pred'] = act_pred
                _, w, h = feature_map.size()
                self.curr_env['attention_map'] = F.interpolate(feature_map.view(1, 1, w, h), (self.img_dim, self.img_dim), mode = 'bilinear').view(1, 72, 72)

            avg_reward = 0
            while(self.above_thresh == False and self.deg_violation == False and self.steps_violation == False):

                self.curr_env['state'] = self.nn_state_space()
                label_value = self.curr_env['label'].float().reshape(1).reshape(-1,1)
                _, _, set_indices = self.map_actions_to_dims()
                # set_indices = self.r_all_action_indices(curr_actions)
                action, log_value, mean_value, pred_agent, critic_val, pred_reward = agent_model.validate(self.curr_env['state'], set_indices, label_value, replay_memory)
                idx, _ = self.one_dimension_action_state(action)
                avg_reward += pred_reward.item()
                r = [action[0].item(), log_value[0].item(), mean_value[0].item(), idx, pred_agent.item(), critic_val.item(), self.curr_env['curr_step'], self.curr_env['name_img']]
                detailed_results.append(r)
                self.curr_env['curr_step'] += 1

            self.aggregate_map_acc(self.saving_dirs['testing_dir'], episode, batch_idx, 'testing')

            some_results = [self.curr_env['name_img'], self.curr_env['label'].item(), self.curr_env['model_pred'].item(), self.curr_env['agg_pred'], self.curr_env['curr_step'], self.curr_env['curr_dims'], (avg_reward)]
            
            overview_results.append(some_results)
            self.reset_episode()

        return (detailed_results, overview_results) 


    def train(self, total_episodes):
        self.agent.get_train()

        for episode in tqdm(range(1, total_episodes+1)):
            start_ep = time.time()
            all_information = []

            for batch_idx, (set_imgs, set_labels, set_names) in tqdm(enumerate(self.training_loader), total=len(self.training_loader)):
                for index, img in enumerate(set_imgs):
                    self.curr_env['img'] = img[None, :]
                    self.curr_env['label'] = set_labels[index]
                    self.curr_env['name_img'] = set_names[index]
                    with torch.no_grad():
                        act_pred, _, feature_map, _ = self.loaded_model(self.curr_env['img'].cuda())
                        self.curr_env['model_pred'] = act_pred
                        _, w, h = feature_map.size()
                        self.curr_env['attention_map'] = F.interpolate(feature_map.view(1, 1, w, h), (self.img_dim, self.img_dim), mode = 'bilinear').view(1, 72, 72)

                    while(self.above_thresh == False and self.deg_violation == False and self.steps_violation == False):
                        self.curr_env['state'] = self.nn_state_space()
                        label_value = self.curr_env['label'].float().reshape(1).reshape(-1,1)
                        _, _, set_indices = self.map_actions_to_dims()
                        curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step, set_indices = self.agent.train(self.curr_env['state'], label_value, set_indices, episode)
                        self.curr_env['curr_step'] += 1
                        transition = ( curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step, set_indices)
                        self.agent.rm.add(transition)

                    set_information = self.agent.optimize()
                    if set_information != None:
                        all_information.append(set_information)

                    self.aggregate_map_acc(self.saving_dirs['training_dir'], episode, batch_idx, 'train')
                    self.saved_information['labels'].append(self.curr_env['label'].item())
                    self.saved_information['model_pred'].append(self.curr_env['model_pred'].item())
                    self.saved_information['agg_pred'].append(self.curr_env['agg_pred'])
                    self.saved_information['img_name'].append(self.curr_env['name_img'])
                    self.reset_episode()
            end_ep = time.time()
            self.save_information(episode, all_information)
            self.reset_episode()
            total_episode_time = ((end_ep - start_ep) / 60, 4)
            print('total_episode_time', total_episode_time)

    
    def train_parallelized(self, total_episodes):
        self.agent.get_train()

        for episode in tqdm(range(1, total_episodes+1)):
            another_time = time.time()
            set_transitions = Parallel(n_jobs=int(16), verbose=0)(delayed(self.single_batch_training)(set_imgs, set_labels, set_names, batch_idx, episode) for batch_idx, (set_imgs, set_labels, set_names) in tqdm(enumerate(self.training_loader), total=len(self.training_loader)))
            self.agent.rm.add_parallel(set_transitions)
            
            print('Optimization Started')
            optim_start = time.time()
            all_information = []
            for i in tqdm(range(800)):
                set_information = self.agent.optimize()
                if set_information != None:
                    all_information.append(set_information)
            optim_end = time.time()

            optimization_time = round((optim_end - optim_start)/60, 4)
            print('Optimization_time: ', optimization_time)

            self.agent.save_models(f"training_{episode}")

            self.save_information(episode, all_information)
            self.reset_episode()
            end_another_time = time.time()
            episode_time = round((end_another_time - another_time) / 60, 4)
            print('Episode time:', episode_time)

    def validate(self, agent_model, episode, replay_memory):
        agent_model.eval()
        
        start_time = time.time()

        all_results = Parallel(n_jobs=int(12), verbose=0)(delayed(self.single_batch_validation)(agent_model, set_imgs, set_labels, set_names, batch_idx, episode, replay_memory) for batch_idx, (set_imgs, set_labels, set_names) in tqdm(enumerate(self.testing_loader), total=len(self.testing_loader)))

        end_time = time.time()

        validation_time = round((end_time - start_time) / 60, 4)
        print('validation_time:', validation_time)
        df_details_arr, df_m_details_arr = [], []
        for arr in all_results:
            df_details_arr.extend(arr[0])
            df_m_details_arr.extend(arr[1])

        df_details = pd.DataFrame(df_details_arr, columns = ["action", "log_value", "mean_value", "idx", 'pred_agent', 'critic_eval', 'curr_step', 'name_img'])
        df_details.to_csv(f"{self.saving_dirs['testing_dir']}/stats/res_validation_{episode}.csv")

        df_m_details = pd.DataFrame(df_m_details_arr, columns = ['name_img', 'label', 'model_pred', 'agg_pred', 'curr_step', 'curr_dims', 'avg_reward'])
        df_m_details.to_csv(f"{self.saving_dirs['testing_dir']}/stats/res_validation_{episode}_details.csv")

        self.reset_episode()


    def save_information(self, episode, all_information):
        df_critic_loss = pd.DataFrame(all_information, columns = ["critic_loss", "actor_loss", "alpha_loss"])

        df_critic_loss.to_csv(f"{self.saving_dirs['training_dir']}/stats/actor_critic_{episode}.csv")


    def check_actions_exists(self, action):
        curr_idx, _ = self.one_dimension_action_state(action[0])

        for p_action in self.curr_env['set_actions'][:-1]:
            indx, _ = self.one_dimension_action_state(p_action[0])
            if indx == curr_idx:
                return True
            
        else:
            return False

    def r_all_action_indices(self, set_actions):
        set_indices = np.zeros(self.len_indices)
        if len(set_actions) > 1:
            for action in set_actions:
                print('action', action)
                idx, _ = self.one_dimension_action_state(action[0])
                set_indices[idx] = 1

        return set_indices

    def step(self, action, evaluate=False):
        self.curr_env['set_actions'].append(action)
        curr_state = self.curr_env['state']
        idx, _ = self.one_dimension_action_state(action[0])
        new_state = self.nn_state_space()

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
             

        if self.check_actions_exists(action):
            self.curr_env['uniqueness_counter'] += 1
            reward = torch.tensor(-0.2)

        if idx >= (self.len_indices-1):
            print('stop_action', idx)
            done = True
            self.above_thresh = True
            reward = torch.tensor(0)

        if self.curr_env['curr_step'] >= self.nb_steps or self.curr_env['uniqueness_counter'] >= self.uniqueness_counter:
            done, self.steps_violation = True, True

        self.curr_env['set_rewards'].append(reward.cpu().item())
        return curr_state, action, reward, new_state, done, pred_test, self.curr_env['label'], self.curr_env['curr_step'], set_indices
