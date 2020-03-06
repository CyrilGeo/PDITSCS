import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
import os


# Replay buffer class
class ReplayBuffer:

    def __init__(self, mem_size, nb_inputs):
        self.memSize = mem_size
        self.memCnt = 0
        self.stateMem = np.zeros((mem_size, nb_inputs))
        self.newStateMem = np.zeros((mem_size, nb_inputs))
        self.actionMem = np.zeros(mem_size, dtype=np.int8)
        self.rewardMem = np.zeros(mem_size)
        # 1 if the state of the transition corresponds to the last state of an episode, 0 otherwise
        self.terminalMem = np.zeros(mem_size)

    # Stores a transition
    def store(self, state, action, reward, new_state, done):
        index = self.memCnt % self.memSize
        self.stateMem[index] = state
        self.actionMem[index] = action
        self.rewardMem[index] = reward
        self.newStateMem[index] = new_state
        self.terminalMem[index] = 1 - int(done)
        self.memCnt += 1

    # Randomly samples a set of transitions of size batch_size
    def sample(self, batch_size):
        nb_samples = min(self.memSize, self.memCnt)
        indices = np.random.choice(nb_samples, batch_size)
        states = self.stateMem[indices]
        actions = self.actionMem[indices]
        rewards = self.rewardMem[indices]
        new_states = self.newStateMem[indices]
        dones = self.terminalMem[indices]
        return states, actions, rewards, new_states, dones


# Q-Network class
class QNetwork(nn.Module):

    def __init__(self, alpha, nb_inputs, nb_actions, fc1_dims, fc2_dims):
        super(QNetwork, self).__init__()
        self.nbInputs = nb_inputs
        self.nbActions = nb_actions
        self.fc1Dims = fc1_dims
        self.fc2Dims = fc2_dims
        self.fc1 = nn.Linear(self.nbInputs, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, self.nbActions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        # self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# Q-Learning agent class
class Agent:

    def __init__(self, alpha, gamma, epsilon, epsilon_end, nb_decay_steps, batch_size, nb_inputs, nb_actions, mem_size,
                 file_name):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonEnd = epsilon_end
        self.decayStep = epsilon / nb_decay_steps
        self.batchSize = batch_size
        self.nbInputs = nb_inputs
        self.nbActions = nb_actions
        self.actionSpace = [x for x in range(nb_actions)]
        self.memSize = mem_size
        self.fileName = file_name
        self.replayBuffer = ReplayBuffer(mem_size, nb_inputs)
        self.qNetwork = QNetwork(alpha, nb_inputs, nb_actions, 512, 512)
        self.targetNetwork = copy.deepcopy(self.qNetwork)

    # Stores a transition form the simulator into the replay buffer
    def store(self, state, action, reward, new_state, done):
        self.replayBuffer.store(state, action, reward, new_state, done)

    # Chooses the next action to be executed by the simulator using epsilon-greedy policy
    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actionSpace)
        else:
            state = T.tensor(state, dtype=T.float32).to(self.qNetwork.device)
            actions = self.qNetwork.forward(state).to(self.qNetwork.device)
            action = T.argmax(actions).item()
        return action

    # Updates the weights of the target net with the weights of the q-net
    def update_target_net(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    # Executes one step of learning (corresponds to one step in the simulator)
    def learning_step(self):
        # Checking if replay buffer is filled enough to sample a batch
        if self.replayBuffer.memCnt < self.batchSize:
            return

        self.qNetwork.optimizer.zero_grad()

        # Samples a batch of transitions from the replay buffer
        states, actions, rewards, new_states, dones = self.replayBuffer.sample(self.batchSize)
        states = T.tensor(states, dtype=T.float32).to(self.qNetwork.device)
        new_states = T.tensor(new_states, dtype=T.float32).to(self.qNetwork.device)

        # Computing q-values
        q_values = self.qNetwork.forward(states).to(self.qNetwork.device)
        q_next_value = self.targetNetwork.forward(new_states).to(self.qNetwork.device)
        q_next_action = self.qNetwork.forward(new_states).to(self.qNetwork.device)

        # Computing target values
        action_selection = T.argmax(q_next_action, dim=1)
        max_q_next = T.tensor([q_next_value[x, action_selection[x]] for x in range(self.batchSize)])
        target_values = q_values.clone()
        batch_indices = np.arange(self.batchSize, dtype=np.int32)
        # Only worth rewards if last state of an episode
        t_val = T.tensor(rewards) + self.gamma * max_q_next * dones
        target_values[batch_indices, actions] = t_val.float().to(self.qNetwork.device)

        # Performs gradient descent
        loss = self.qNetwork.loss(target_values, q_values).to(self.qNetwork.device)
        loss.backward()
        self.qNetwork.optimizer.step()

        # Updates epsilon
        self.epsilon = self.epsilon - self.decayStep if self.epsilon > self.epsilonEnd else self.epsilonEnd

    def save_net(self):
        T.save(self.qNetwork.state_dict(), os.path.join("models", self.fileName))

    def load_net(self):
        self.qNetwork.load_state_dict(T.load(os.path.join("models", self.fileName)))
