import numpy as np
import random
from collections import deque
import torch
import pandas as pd
from tqdm import tqdm
import time

class RunningStats:
    def __init__(self, num_features):
        self.num_features = num_features
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.zeros(num_features)
        self.count = 0

    def update(self, observations):
        batch_mean = torch.mean(observations, dim=0)
        batch_var = torch.var(observations, dim=0)

        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            delta = batch_mean - self.running_mean
            total_count = self.count + observations.size(0)

            # Update running mean
            self.running_mean += delta * observations.size(0) / total_count

            # Update running variance
            m_a = self.running_var * self.count
            m_b = batch_var * observations.size(0)
            M2 = m_a + m_b + torch.square(delta) * self.count * observations.size(0) / total_count
            self.running_var = M2 / total_count


        self.count += observations.size(0)

    def define_running_stats(self, running_mean:np.array, running_var:np.array):
        self.running_mean = torch.from_numpy(running_mean)
        self.running_var = torch.from_numpy(running_var)


    def get_mean(self):
        return self.running_mean

    def get_std(self):
        return torch.sqrt(self.running_var)

class ReplayMemory:

    def __init__(self, size, num_features):
        self.buffer = deque(maxlen= size)
        self.maxSize = size
        self.len = 0
        self.running_stats = RunningStats(num_features)

    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        curr_state  = np.float32([arr[0] for arr in batch])
        a_arr  = np.float32([arr[1] for arr in batch])
        r_arr  = np.float32([arr[2] for arr in batch])
        next_state_arr  = np.float32([arr[3] for arr in batch])
        terminal_arr  = np.array([arr[4] for arr in batch])
        pred_test_arr  = np.float32([arr[5] for arr in batch])
        labels_arr  = np.float32([arr[6] for arr in batch])
        curr_step_arr  = np.float32([arr[7] for arr in batch])
        set_indices_arr  = np.float32([arr[8] for arr in batch])



        curr_state = curr_state.reshape(-1, 1375)

        next_state_arr = next_state_arr.reshape(-1,1375)

        labels_arr = labels_arr.reshape(-1, 1)

        return curr_state, a_arr, r_arr, next_state_arr, terminal_arr, pred_test_arr, labels_arr, curr_step_arr, set_indices_arr
    

    def len(self):
        return self.len
    
    def update_running_stats(self, running_mean: np.array, running_var:np.array):
        self.running_stats.define_running_stats(running_mean, running_var)

    def add_parallel(self, transitions):
        self.len += len(transitions) * len(transitions[0])

        print('started the addition process')
        if self.len > self.maxSize:
            self.len = self.maxSize
        
        start_time = time.time()
        for idx in tqdm(range(len(transitions))):
            # set_states = torch.Tensor()
            transition = transitions[idx]
            for single_transition in transition:
                self.buffer.append(single_transition)

        end_time = time.time()
        total_time = np.round((end_time - start_time)/ 60, 4)
        print('Addition time:', total_time)

    def add(self, transitions):
        # transition = (state, action, reward, next_state, terminal, pred_test, label, curr_step)
        self.len += 1

        # print('started the addition process')
        if self.len > self.maxSize:
            self.len = self.maxSize
        
        self.buffer.append(transitions)
        
        
    def save(self, saving_dir):
        running_mean = self.running_stats.running_mean.numpy()
        running_var = self.running_stats.running_var.numpy()

        df_saved = pd.DataFrame()
        df_saved['mean'] = running_mean
        df_saved['var'] = running_var

        df_saved.to_csv(f"{saving_dir}")

