# from replay_memory_int import ReplayMemory
from grid_models.replay_memory import ReplayMemory
from Utils.actor_critic_utils import *
from grid_models.actor_critic_models import *

class Actor_Trainer:

    def __init__(self, target_episode, img_dim, state_dim, action_dim, size_buffer, batch_size, lr_a, rl_env, saving_dir, discount_factor):
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

        self.rm = ReplayMemory(self.size_buffer)
        self.utils = Utils([self.img_dim, self.img_dim])
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = ImageActor(self.action_dim)
        self.target_actor = ImageActor(self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr_a)
        self.utils.hard_update(self.target_actor, self.actor)

        self.critic = ImageCritic(self.action_dim)
        self.target_critic = ImageCritic(self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr_a*10)
        self.utils.hard_update(self.target_critic, self.critic)
        self.nb_elements = 0

        
        self.saved_information = {
            'critic_loss': [],
            'actor_loss': [],
            'actions': [],
            'reward': [],
            'pred_test': [],
            'label': [],
            'curr_step': []
        }
        self.set_rewards = []

        self.cuda()
        self.get_train()

    def cuda(self):
        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def get_train(self):
        self.actor.train()
        self.target_actor.train()
        self.critic.train()
        self.target_critic.train()

    def eval(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_target_actor.pt")
        torch.save(self.target_critic.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_target_critic.pt")

        torch.save(self.actor.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_actor.pt")
        torch.save(self.critic.state_dict(), f"{self.saving_dir['training_dir']}/models/EP-{episode_count}_critic.pt")

        print("Models saved successfully")
    

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(f"{self.saving_dir}/models/EP-training_{episode}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"{self.saving_dir}/models/EP-training_{episode}_critic.pt"))

        self.utils.hard_update(self.target_actor, self.actor)
        self.utils.hard_update(self.target_critic, self.critic)

        print("Models loaded succesfully")

    # This function returns the exploitative action from the target_actor
    def get_exploitation(self, state, label):
        state = Variable(state)
        with torch.no_grad():
            action = self.target_actor(state, label).detach().cpu()[0]

        return action.data.numpy()

    # This function defines the agent's exploration strategy    
    def get_exploration_action(self, state, label):
        state = Variable(state)
        with torch.no_grad():
            action = self.target_actor(state, label).detach().cpu()[0]
            noise_sample = self.noise.sample()
            new_action = action.data.numpy() + noise_sample
            new_set_actions = []
            for a in range(self.action_dim):
                new_set_actions.append(np.clip(new_action[a], 0, 1))

        return new_set_actions
    
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
    
    # This function defines the core agent's convergence
    def optimize(self):
        
        if self.rm.len < (self.size_buffer):
            return
        curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step_arr = self.rm.sample(self.batch_size)

        self.set_rewards.append(pred_reward)
        
        state = torch.from_numpy(curr_state).cuda()
        next_state = torch.from_numpy(new_state).cuda()
        set_labels = torch.from_numpy(label).cuda()

        action = torch.from_numpy(action).cuda()
        reward = torch.from_numpy(pred_reward).cuda()
        terminal = torch.from_numpy(done).cuda()
            
        # ----- optimize critic ------ #
        a_pred = self.target_actor(next_state, set_labels)
        target_values = torch.add(reward, torch.mul(~terminal, torch.mul(self.discount_factor, self.target_critic(next_state, set_labels, a_pred).reshape(-1,))))
        # print(f'reward: {reward}, target_critic: {self.target_critic(next_state, set_labels, a_pred)}')
        val_expected = self.critic(state, set_labels, action).reshape(-1,)

        # print(f'a_pred: {a_pred}, target_values: {target_values}, val_expected: {val_expected}')
        
        criterion = nn.MSELoss()
        loss_critic = criterion(target_values, val_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
            
        # ----- optimize actor ----- #
        pred_a1 = self.actor(state, set_labels)
        loss_actor = -self.critic(state, set_labels, pred_a1).mean()

        # for name, param in self.actor.named_parameters():
        #     print(f'name: {name}, param.grad: {param.grad}')
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        self.saved_information['critic_loss'].append(loss_critic.item())
        self.saved_information['actor_loss'].append(loss_actor.item())
        self.saved_information['actions'].append(action.cpu().numpy())
        self.saved_information['reward'].append(reward.cpu().numpy())
        self.saved_information['pred_test'].append(pred_test)
        self.saved_information['label'].append(label)
        self.saved_information['curr_step'].append(curr_step_arr)

        print(f'loss_critic: {loss_critic.item()}, loss_actor: {loss_actor.item()}, terminal: {terminal}')
        TAU = 0.001

        self.utils.soft_update(self.target_actor, self.actor, TAU)
        self.utils.soft_update(self.target_critic, self.critic, TAU)


    
    def normalize_rewards(self, reward_val):
        
        returned_value = reward_val

        purge_value = 1000
        if len(self.set_rewards) >= 2 and self.nb_elements <= purge_value:
            rewards_arr = np.array(self.set_rewards)
            mean_rewards = np.mean(rewards_arr)
            std_rewards = np.std(rewards_arr)
            self.mean_rewards = mean_rewards
            self.std_rewards = std_rewards

            returned_value = (reward_val - mean_rewards) / (std_rewards + 1e-5)

        elif len(self.set_rewards) > 2 and self.nb_elements > purge_value:
            returned_value = (reward_val - self.mean_rewards) / (self.std_rewards + 1e-5)

        if len(self.set_rewards) >= purge_value:
            rewards_arr = np.array(self.set_rewards)
            mean_rewards = np.mean(rewards_arr)
            std_rewards = np.std(rewards_arr)
            self.mean_rewards = ((self.nb_elements - purge_value) / (self.nb_elements)) * (self.mean_rewards) + ((purge_value) / (self.nb_elements)) * (mean_rewards) 
            self.std_rewards = ((self.nb_elements - purge_value) / (self.nb_elements)) * (self.std_rewards) + ((purge_value) / (self.nb_elements)) * (std_rewards) 
            self.set_rewards = []

        return returned_value
    

    # This defines the training process
    def train(self, state, label, episode):
        state = state[:, None]
        state = state.cuda()
        label = label.cuda()
        action = None
        if episode < self.target_episode:
            action = self.get_exploration_action(state, label)
        else:
            action = self.get_exploitation(state, label)

        curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step = self.rl_env.step(action)

        self.nb_elements += 1
        # reward = self.normalize_rewards(pred_reward.item())
        self.saved_information['reward'].append(pred_reward)

        self.rm.add(curr_state.detach().cpu().numpy(), action, pred_reward.item(), new_state.detach().cpu().numpy(), done, pred_test.item(), label, curr_step)
        self.optimize()


    def validate(self, state, label):

        state = state.cuda()
        label = label.cuda()
        
        state = state[None, :]

        label = label[None, :]
        action = self.get_exploitation(state, label)
        curr_state, action, pred_reward, new_state, done, pred_test, label, curr_step = self.rl_env.step(action)

        state = curr_state[None, :]
        action = action[None, :]
        label = label.reshape(1, -1).float()
        action = torch.from_numpy(action).cuda()
        state = state.cuda()
        label = label.cuda()

        with torch.no_grad():
            critic_val = self.critic(state, label, action)

        return action, pred_test, pred_reward, critic_val
        # reward = (reward.item() - self.mean_rewards) / (self.std_rewards + 1e-5)

        # Here, we need to define the type of saved information
        # self.saved_information['diff_reward'].append(diff_reward)
        # self.saved_information['critic_loss'].append(critic_val.item())
        # self.saved_information['actions'].append(action)
        # self.saved_information['reward'].append(reward)
        # self.saved_information['pred_test'].append(pred_test.item())
        # self.saved_information['int_penalty'].append(int_penalty)
        # self.saved_information['label'].append(label)

    


