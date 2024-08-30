from grid_models.sac_models import *
from torch.autograd import Variable
import numpy as np
from grid_models.replay_memory import ReplayMemory
from Utils.actor_critic_utils import *
import time
import pandas as pd

# Some of the code is adopted from https://github.com/pranz24/pytorch-soft-actor-critic/tree/master
class Sac_Trainer():

    def __init__(self, target_episode, img_dim, state_dim, action_dim, size_buffer, batch_size, lr_a, rl_env, saving_dir, discount_factor,
                 alpha, grad_clip, device):
        self.target_episode = target_episode # This defines the episode, whereby the agent starts exploiting its exploration-based knowledge
        self.state_dim = state_dim # The state dimenions
        self.action_dim = action_dim # This variable stores the action dimension
        self.size_buffer = size_buffer # This variable defines the size of the replay memory
        self.batch_size = batch_size
        self.lr_a = lr_a # The agent's learning rate
        self.rl_env = rl_env # This defines the Environment that the RL agent interacts with
        self.saving_dir = saving_dir # This variable stores the directory where the information is stored
        self.img_dim = img_dim
        self.discount_factor = discount_factor

        self.rm = ReplayMemory(self.size_buffer, 1375)
        self.utils = Utils([self.img_dim, self.img_dim])
        

        self.policy = GaussianNNPolicy(1375, self.action_dim, rl_env.len_indices)
        self.critic = TwinnedQNetwork(1375, action_dim, rl_env.len_indices)
        self.target_critic = TwinnedQNetwork(1375, action_dim, rl_env.len_indices)
        self.started_training = False
        self.nb_elements = 0
        self.set_rewards = []
        self.device = device
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device = self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr = lr_a)
        self.alpha = alpha

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr = self.lr_a / 10)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.lr_a)
        
        self.grad_clip = grad_clip

        self.saved_information = {
            'critic_loss': [],
            'actor_loss': [],
            'actions': [],
            'reward': [],
            'pred_test': [],
            'label': [],
            'curr_step': []
        }

        self.utils.hard_update(self.target_critic, self.critic)
        # self.cuda()
        self.set_to_device(self.device)
        self.get_train()

    def reset_information(self):
        self.saved_information = {
            'critic_loss': [],
            'actor_loss': [],
            'actions': [],
            'reward': [],
            'pred_test': [],
            'label': [],
            'curr_step': []
        }

    def save_models(self, episode_count):
        torch.save(self.target_critic.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_target_critic.pt")
        torch.save(self.policy.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_actor.pt")
        torch.save(self.critic.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_critic.pt")
        self.rm.save(f"{self.saving_dir['training_dir']}/rm-{episode_count}.csv")

        print("Models saved successfully")
    
    def cuda(self):
        self.policy.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def set_to_device(self, device):
        self.policy = self.policy.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)


    def get_train(self):
        self.policy.train()
        self.critic.train()
        self.target_critic.train()

    def eval(self):
        self.policy.eval()
        self.critic.eval()
        self.target_critic.eval()

    def load_models(self, episode, evaluate = True):
        self.policy.load_state_dict(torch.load(f"{self.saving_dir['training_dir']}/models/EP-training_{episode}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"{self.saving_dir['training_dir']}/models/EP-training_{episode}_critic.pt"))
        self.utils.hard_update(self.target_critic, self.critic)
        
        df_r = pd.read_csv(f"{self.saving_dir['training_dir']}/rm-training_{episode}.csv")
        self.rm.update_running_stats(df_r['mean'].values, df_r['var'].values)

        if evaluate:
            self.eval()
        else:
            self.get_train()

        print("Models loaded succesfully")
        

    def get_exploration_action(self, state, label, set_indices):
        state = Variable(state)
        set_indices = Variable(torch.from_numpy(set_indices)).cuda()
        set_indices = set_indices.reshape(-1, len(set_indices))
       
        with torch.no_grad():
            action, log_prob, mean = self.policy.sample(state, label, set_indices)

        return action.cpu().numpy().reshape(-1)
    
    def get_exploitation(self, state, label, set_indices):
        # acting without randomness
        state = Variable(state)
        set_indices = Variable(torch.from_numpy(set_indices)).cuda()
        set_indices = set_indices.reshape(-1, len(set_indices))

        with torch.no_grad():
            action, log_prob, mean = self.policy.sample(state, label, set_indices)

        return action.cpu().numpy().reshape(-1), log_prob.cpu().numpy().reshape(-1), mean.cpu().numpy().reshape(-1) 
    
    def optimize(self):
        
        set_information = []
        if self.rm.len < (self.size_buffer):
            return

        self.started_training = True
        curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step_arr, set_indices_arr = self.rm.sample(self.batch_size)

        self.set_rewards.append(pred_reward)
        
        # state = torch.from_numpy(curr_state).to(self.device)
        state = torch.from_numpy(curr_state).to(self.device)

        next_state = torch.from_numpy(new_state).to(self.device)
        set_labels = torch.from_numpy(label).to(self.device)

        action = torch.from_numpy(action).to(self.device)
        reward = torch.from_numpy(pred_reward).to(self.device)
        terminal = torch.from_numpy(done).to(self.device)
        set_indices_t = torch.from_numpy(set_indices_arr).to(self.device)
        new_set_indices = []
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state, set_labels, set_indices_t)
            for idx, n_set in enumerate(set_indices_arr):
                action_idx = next_state_action[idx]
                idx_nbr, _ = self.rl_env.one_dimension_action_state(action_idx)
                n_set[idx_nbr] = 1
                new_set_indices.append(n_set)
            new_set_indices = torch.from_numpy(np.array(new_set_indices)).to(self.device)
            
            qf1_next_target, qf2_next_target = self.target_critic(next_state, set_labels, next_state_action, new_set_indices)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + torch.mul(~terminal, torch.mul(self.discount_factor, min_qf_next_target).T)
            next_q_value = next_q_value.permute(1,0)
        
        qf1, qf2 = self.critic(state, set_labels, action, set_indices_t)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = (qf1_loss + qf2_loss)

        self.critic_optimizer.zero_grad()
        
        qf_loss.backward()

        self.critic_optimizer.step()

        pi, log_pi, _ = self.policy.sample(state, set_labels, set_indices_t)
        new_set_indices = []
        for idx, n_set in enumerate(set_indices_arr):
            action_idx = pi[idx]
            idx_nbr, _ = self.rl_env.one_dimension_action_state(action_idx)
            n_set[idx_nbr] = 1
            new_set_indices.append(n_set)
        new_set_indices = torch.from_numpy(np.array(new_set_indices)).to(self.device)
        qf1_pi, qf2_pi = self.critic(state, set_labels, pi, new_set_indices)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        

        self.optimizer_policy.step()

        TAU = 0.001

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        set_information = [qf_loss.item(), policy_loss.item(), alpha_loss.item()]
        self.utils.soft_update(self.target_critic, self.critic, TAU)

        return set_information
    
    def train(self, state, label, set_indices_arr, episode):
        # state = state[:, None].to(self.device)
        state = state.to(self.device)
        label = label.to(self.device)
        action = None
        self.alpha = self.alpha / episode


        if episode < self.target_episode:
            action = self.get_exploration_action(state, label, set_indices_arr)
        else:
            action, _, _ = self.get_exploitation(state, label, set_indices_arr)

        curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step, set_indices = self.rl_env.step(action, evaluate=False)
        self.nb_elements += 1
        self.saved_information['reward'].append(pred_reward)


        return curr_state.detach().cpu().numpy(), action, pred_reward.item(), new_state.detach().cpu().numpy(), done, pred_test.item(), label, curr_step, set_indices


    def validate(self, state, set_indices_arr, label, replay_memory:ReplayMemory):
        # state = state[None, :].to(self.device)
        # state = (state.cpu() - replay_memory.running_stats.get_mean()) / (replay_memory.running_stats.get_std())

        state = state.to(self.device)

        label = label.to(self.device)

        action, log_value, mean_value = self.get_exploitation(state, label, set_indices_arr)
        curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step, set_indices = self.rl_env.step(action, evaluate=True)

        # state = curr_state[None, :].to(self.device)
        state = curr_state.to(self.device)

        # mean_value = mean_value[None, :]
        action = action[None, :]

        label = label.reshape(1, -1).float()
        label = label.to(self.device)
        action = torch.from_numpy(action).to(self.device)
        
        set_indices = Variable(torch.from_numpy(set_indices)).cuda()
        set_indices = set_indices.reshape(-1, len(set_indices))

        with torch.no_grad():
            qf1, qf2 = self.critic(state, label, action, set_indices)
            critic_val = torch.min(qf1, qf2)

        return action, log_value, mean_value, pred_test, critic_val, pred_reward



    

