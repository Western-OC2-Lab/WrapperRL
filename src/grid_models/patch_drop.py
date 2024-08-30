from Utils.actor_critic_utils import *
from grid_models.sac_models import *
from tqdm import tqdm
from torch.distributions import Multinomial, Bernoulli
from CNN_model.constants import reverse_image_transform
import os
from matplotlib import pyplot as plt
import pandas as pd

class PatchDrop():


    def __init__(self, loaded_model, target_episode, img_dim, state_dim, lr, batch_size, saving_dir,
                 alpha, device, grid_size, training_loader, validation_loader):
        
        self.target_episode = target_episode
        
        self.img_dim = img_dim
        self.state_dim = state_dim
        self.saving_dir = saving_dir
        self.alpha = alpha
        self.device = device
        self.loaded_model = loaded_model
        self.loaded_model = self.loaded_model.to(self.device)
        self.loaded_model.eval()
        self.grid_size = grid_size

        self.lr = lr
        self.batch_size = batch_size
        self.utils = Utils([self.img_dim, self.img_dim])
        self.nb_cols, self.nb_rows = (self.img_dim // grid_size[0]), (self.img_dim // grid_size[1])
        self.len_indices = self.nb_cols * self.nb_rows
        self.action_dim = self.len_indices

        self.agent = PatchDropActor(1375, self.len_indices)
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.cuda()

        self.set_rewards = []
        self.alpha = alpha
        self.agent_optimizer = optim.Adam(self.agent.parameters(), lr = self.lr)
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


    def save_models(self, episode_count):
        torch.save(self.agent.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_patchDrop.pt")

        print("Models saved successfully")


    def cuda(self):
        self.agent.cuda()

    def set_to_device(self, device):
        self.agent = self.agent.to(device)

    def get_train(self):
        self.agent.train()

    def eval(self):
        self.agent.eval()

    def load_models(self, episode, evaluate = True):
        self.agent.load_state_dict(torch.load(f"{self.saving_dir['training_dir']}/models/EP-training_{episode}_patchDrop.pt"))
        

        if evaluate:
            self.eval()
        else:
            self.get_train()

        print("Models loaded succesfully")

    def nn_state_space(self, curr_img):
        
        curr_img = curr_img[None, :]
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output[0]
            return hook
        self.loaded_model.attn2.register_forward_hook(get_activation('attn2'))

        _, _, attn_part, _ = self.loaded_model(curr_img.cuda())

        fc_image = activation['attn2']
        B, C, H, W = fc_image.size()
        feature_part = fc_image.reshape(B, C*H*W)
        B, H, W = attn_part.size()

        attn_part = attn_part.reshape(B, H*W)

        final_state = torch.cat((feature_part, attn_part), dim = 1)

        return final_state
    
    def map_actions_to_dims(self, set_indices):
        set_actions = []
        for idx, grid_position in enumerate(set_indices):
            if grid_position == 1:
                row_idx, col_idx = idx // (self.nb_rows), idx % (self.nb_cols)
                X_start, Y_start = (col_idx * self.grid_size[0]), (row_idx * self.grid_size[0])

                values = [X_start, self.grid_size[0], Y_start, self.grid_size[0]]
                set_actions.append(values)

        return set_actions
    
    def compute_reward(self, preds, targets, policy, penalty):

        patch_use = policy.sum(1).float() / policy.size(1)
        sparse_reward = 1.0 - patch_use ** 2
        # _, pred_idx = preds.max(1)
        preds = torch.round(preds)
        preds = torch.squeeze(preds)

        match = (preds==targets).data

        reward = sparse_reward
        reward[~match] = penalty
        reward = reward.unsqueeze(1)

        return reward, match.float()
    
    def transform_image(self, tensor_array):
        tensor_array = torch.mul(tensor_array, 255).numpy()
        instance_arr = (tensor_array + 1) /2
        instance_arr = tensor_array
        instance_arr[instance_arr > 1.0] = 1.0
        instance_arr[instance_arr < 0.0] = 0.0
        instance_arr = np.array(instance_arr, np.float32) * 255
        instance_arr = instance_arr.astype(np.uint8)

        return instance_arr
    
    def plot_predictions(self, saving_dir, r_weights, pred_test, orig_img, name_img, label, ep, batch, actions):
        name_img = name_img.split("/")[-1]
        fig, ax = plt.subplots(1, 2, figsize = (15, 15))
        axes = ax.flatten()


        plotted_values_new = self.transform_image(r_weights[0, :, :, :].permute(1,2, 0))
        plotted_values_img = self.transform_image(orig_img[0, :, :, :].permute(1, 2, 0))
        
        axes[0].imshow(plotted_values_new)
        axes[1].imshow(plotted_values_img)
        plt.suptitle(f"Pred: {np.round(pred_test.item(), 3)}, Label: {label.item()}, Nb_steps: {len(actions)}", fontsize = 35)

        plt.tight_layout()
        plt.savefig(f"{saving_dir}/{name_img}_{ep}_{batch}.png", pad_inches = 0, bbox_inches = 'tight')
        plt.close()
    
    def aggregate_map_acc(self, saving_dir, set_actions, set_imgs, set_names, set_labels, ep, batch_idx, type_val = 'train'):
        freq = 500 if type_val == 'train' else 25
        if batch_idx % freq == 0:
            saving_dir = f"{saving_dir}/images/ep_{ep}"
            os.makedirs(saving_dir, exist_ok=True)
            for idx, actions in enumerate(set_actions):
                selected_img, _ = self.utils.select_region(set_imgs[idx], actions)
                pred_test, _, _, _ = self.loaded_model(selected_img.cuda())

                # orig_img = reverse_image_transform(set_imgs[idx])
                # print('actions', actions)
                # s_image = self.utils.select_region(orig_img, actions)
                # orig_img = orig_img[None, :]
                curr_img = set_imgs[idx]
                r_weights, orig_img = self.utils.select_region(reverse_image_transform(curr_img),actions, interpolate=True)

                self.plot_predictions(saving_dir, r_weights, pred_test, orig_img, set_names[idx], set_labels[idx], ep, batch_idx, actions)

    def train(self, epoch):
        self.get_train()

        for batch_idx, (set_imgs, set_labels, set_names) in tqdm(enumerate(self.training_loader), total=len(self.training_loader)):

                set_states = torch.Tensor().cuda()
                for img in set_imgs:
                    state = self.nn_state_space(img)
                    set_states = torch.cat((set_states, state), dim = 0)
                output_agent = self.agent(set_states)
                probs = torch.sigmoid(output_agent)
                probs = probs * self.alpha + (1-self.alpha) * (1-probs)

                dist = Bernoulli(probs)
                policy_samples = dist.sample()

                policy_maps = probs.data.clone()
                policy_maps[policy_maps<0.5] = 0.0
                policy_maps[policy_maps>=0.5] = 1.0


                set_actions_samples = [self.map_actions_to_dims(policy_sample) for policy_sample in policy_samples]
                set_actions = [self.map_actions_to_dims(policy_map) for policy_map in policy_maps] 

                img_samples = torch.Tensor().cuda()
                for idx, img in enumerate(set_imgs):
                    img_samples = torch.cat((img_samples, self.utils.select_region(img, set_actions_samples[idx])[0].cuda()), dim = 0)

                img_maps = torch.Tensor().cuda()
                for idx, img in enumerate(set_imgs):
                    img_maps = torch.cat((img_maps, self.utils.select_region(img, set_actions[idx])[0].cuda()), dim = 0)

                preds_sample, _, _, _ = self.loaded_model(img_samples)
                preds_map, _, _, _ = self.loaded_model(img_maps)
                preds_sample = preds_sample.cpu()
                preds_map = preds_map.cpu()
                
                reward_map, match = self.compute_reward(preds_map, set_labels, policy_maps.data, -0.5)
                reward_sample, _ = self.compute_reward(preds_sample, set_labels, policy_samples.int(), -0.5)

                advantage = reward_sample.cuda().float() - reward_map.cuda().float()

                loss = -dist.log_prob(policy_samples)
                loss = loss * Variable(advantage).expand_as(policy_samples)
                loss = loss.mean()
                preds_sample = torch.squeeze(preds_sample)
                loss += F.binary_cross_entropy(preds_sample, set_labels.float())

                self.agent_optimizer.zero_grad()
                loss.backward()
                self.agent_optimizer.step()

                self.aggregate_map_acc(self.saving_dir['training_dir'], set_actions_samples, set_imgs, set_names, set_labels, epoch, batch_idx, type_val = 'train')

        self.save_models(f"training_{epoch}")

    def validate(self, agent_model, episode):
        agent_model.eval()

        set_rewards = []
        set_images = []
        nb_actions  = []
        predictions = []
        labels = []
        for batch_idx, (set_imgs, set_labels, set_names) in tqdm(enumerate(self.validation_loader), total=len(self.validation_loader)):

            set_states = torch.Tensor().cuda()
            for img in set_imgs:
                state = self.nn_state_space(img)
                set_states = torch.cat((set_states, state), dim = 0)
            output_agent = self.agent(set_states)
            probs = torch.sigmoid(output_agent)
            # probs = probs * self.alpha + (1-self.alpha) * (1-probs)

            policy_maps = probs.data.clone()
            policy_maps[policy_maps<0.5] = 0.0
            policy_maps[policy_maps>=0.5] = 1.0


            set_actions = [self.map_actions_to_dims(policy_map) for policy_map in policy_maps] 

            img_maps = torch.Tensor().cuda()
            for idx, img in enumerate(set_imgs):
                img_maps = torch.cat((img_maps, self.utils.select_region(img, set_actions[idx])[0].cuda()), dim = 0)


            preds_map, _, _, _ = self.loaded_model(img_maps)
            preds_map = preds_map.cpu()

            reward, _ = self.compute_reward(preds_map, set_labels, policy_maps.data, -0.5)
            appended_rewards = [r.cpu().item() for r in reward]
            preds_map = preds_map.detach().numpy()
            set_rewards.extend(appended_rewards)
            set_images.extend(set_imgs)
            predictions.extend(preds_map)
            len_actions = [len(action) for action in set_actions]
            nb_actions.extend(len_actions)
            labels.extend(set_labels)
            self.aggregate_map_acc(self.saving_dir['testing_dir'], set_actions, set_imgs, set_names, set_labels, episode, batch_idx, type_val = 'test')

        
        # new_df = pd.DataFrame([set_rewards, set_images, predictions, nb_actions], columns = ["reward", "img", "pred", "actions"])
        new_df = pd.DataFrame(columns = ['img', 'reward', 'pred', 'actions', 'labels'])
        print('imgs', len(set_images))
        print('reward', len(set_rewards))
        print('preds', len(predictions))
        print('actions', len(nb_actions))

        new_df['img'] = set_images
        new_df['reward'] = set_rewards
        new_df['pred'] = predictions
        new_df['actions'] = nb_actions
        new_df['labels'] = labels



        new_df.to_csv(f"{self.saving_dir['testing_dir']}/stats/res_validation_{episode}.csv")