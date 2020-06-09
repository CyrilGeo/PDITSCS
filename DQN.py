"""
Double Q-Learning agent.
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
import os
import sys


class ReplayBuffer:
    """
    The replay buffer used to store state transitions and to draw batches for computing the loss.
    """

    def __init__(self, mem_size, nb_inputs):
        self.memSize = mem_size
        self.memCnt = 0
        self.stateMem = np.zeros((mem_size, nb_inputs))
        self.newStateMem = np.zeros((mem_size, nb_inputs))
        self.actionMem = np.zeros(mem_size, dtype=np.int8)
        self.rewardMem = np.zeros(mem_size)
        # 0 if the state of the transition corresponds to the last state of an episode, 1 otherwise
        self.terminalMem = np.zeros(mem_size)

    def store(self, state, action, reward, new_state, done):
        """
        Stores a transition into the replay buffer.
        :param state: initial state of the transition
        :param action: action  of the transition, performed in the initial state
        :param reward: reward of the transition, for performing the action in the initial state
        :param new_state: new state of the transition, reached after performing the action in the initial state
        :param done: 1 if this transition terminated an episode, 0 otherwise
        :return: None
        """
        index = self.memCnt % self.memSize
        self.stateMem[index] = state
        self.actionMem[index] = action
        self.rewardMem[index] = reward
        self.newStateMem[index] = new_state
        self.terminalMem[index] = 1 - int(done)
        self.memCnt += 1

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions of size batch_size.
        :param batch_size: the batch size
        :return: the batch of transitions
        """
        nb_samples = min(self.memSize, self.memCnt)
        indices = np.random.choice(nb_samples, batch_size)
        states = self.stateMem[indices]
        actions = self.actionMem[indices]
        rewards = self.rewardMem[indices]
        new_states = self.newStateMem[indices]
        dones = self.terminalMem[indices]
        return states, actions, rewards, new_states, dones


class QNetwork(nn.Module):
    """
    The Q-Network used as an approximation of the Q-function.
    """

    def __init__(self, alpha, milestones, lr_decay_factor, nb_inputs, nb_actions, fc1_dims, fc2_dims):
        super(QNetwork, self).__init__()
        self.nbInputs = nb_inputs
        self.nbActions = nb_actions
        self.fc1Dims = fc1_dims
        self.fc2Dims = fc2_dims
        self.fc1 = nn.Linear(self.nbInputs, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, self.nbActions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=lr_decay_factor)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, states):
        """
        Performs a forward computation in the Q-Network.
        :param states: the input state representation
        :return: the output Q-values
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent:
    """
    The Q-Learning agent.
    """

    def __init__(self, alpha, milestones, lr_decay_factor, gamma, policy, epsilon, epsilon_end, nb_decay_steps_ep, temp,
                 temp_end, nb_decay_steps_temp, batch_size, nb_inputs, nb_actions, mem_size, file_name):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        self.policy = policy
        # Epsilon-greedy
        self.epsilon = epsilon
        self.epsilonEnd = epsilon_end
        self.decayStepEpsilon = (epsilon - epsilon_end) / nb_decay_steps_ep
        # Boltzmann exploration
        self.temp = temp
        self.tempEnd = temp_end
        self.decayStepTemp = (temp - temp_end) / nb_decay_steps_temp

        self.batchSize = batch_size
        self.nbInputs = nb_inputs
        self.nbActions = nb_actions
        self.actionSpace = [x for x in range(nb_actions)]
        self.memSize = mem_size
        self.fileName = file_name
        self.replayBuffer = ReplayBuffer(mem_size, nb_inputs)
        self.qNetwork = QNetwork(alpha, milestones, lr_decay_factor, nb_inputs, nb_actions, 512, 512)
        self.targetNetwork = copy.deepcopy(self.qNetwork)

    def store(self, state, action, reward, new_state, done):
        """
        Stores a transition form the simulator into the replay buffer.
        :param state: initial state of the transition
        :param action: action  of the transition, performed in the initial state
        :param reward: reward of the transition, for performing the action in the initial state
        :param new_state: new state of the transition, reached after performing the action in the initial state
        :param done: 1 if this transition terminated an episode, 0 otherwise
        :return: None
        """
        self.replayBuffer.store(state, action, reward, new_state, done)

    def select_action(self, state, no_policy=False):
        """
        Chooses the next action to be executed by the simulator using epsilon-greedy or Boltzmann exploration policy.
        :param state: the state from which the next action to perform is selected
        :param no_policy: determines if the action is selected with or without an exploration policy
        :return: the selected action
        """
        if no_policy:
            state = T.tensor(state, dtype=T.float32).to(self.qNetwork.device)
            q_values = self.qNetwork.forward(state).to(self.qNetwork.device)
            action = T.argmax(q_values).item()
        elif self.policy == "epsilon-greedy":
            if random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(self.actionSpace)
            else:
                state = T.tensor(state, dtype=T.float32).to(self.qNetwork.device)
                q_values = self.qNetwork.forward(state).to(self.qNetwork.device)
                action = T.argmax(q_values).item()
        elif self.policy == "boltzmann":
            state = T.tensor(state, dtype=T.float32).to(self.qNetwork.device)
            q_values = self.qNetwork.forward(state).to(self.qNetwork.device)
            probs = F.softmax(q_values / self.temp, dim=0).to("cpu").detach().numpy()
            action = np.random.choice(self.actionSpace, p=probs)
        else:
            sys.exit("Invalid policy specified. Either choose \"epsilon-greedy\" or \"boltzmann\".")
        return action

    def update_target_net(self):
        """
        Updates the weights of the target network with the weights of the Q-Network.
        :return: None
        """
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    def scheduling_step(self):
        """
        Executes one step of scheduling for the learning rate.
        :return: None
        """
        self.qNetwork.scheduler.step()

    def learning_step(self):
        """
        Executes one step of learning (corresponds to one step in the simulator).
        :return: None
        """
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
        max_q_next = T.tensor([q_next_value[x, action_selection[x]] for x in range(self.batchSize)]).float()
        target_values = q_values.clone()
        batch_indices = np.arange(self.batchSize, dtype=np.int32)
        # Only worth rewards if last state of an episode
        t_val = T.tensor(rewards).float() + self.gamma * max_q_next * T.tensor(dones).float()
        target_values[batch_indices, actions] = t_val.to(self.qNetwork.device)

        # Performs gradient descent
        loss = self.qNetwork.loss(target_values, q_values).to(self.qNetwork.device)
        loss.backward()
        self.qNetwork.optimizer.step()

        # Updates epsilon and temperature
        self.epsilon = self.epsilon - self.decayStepEpsilon if self.epsilon > self.epsilonEnd else self.epsilonEnd
        self.temp = self.temp - self.decayStepTemp if self.temp > self.tempEnd else self.tempEnd

    def save_net(self, file_name=None):
        """
        Saves the Q-Network in a file.
        :param file_name: the file name
        :return: None
        """
        if file_name:
            T.save(self.qNetwork.state_dict(), os.path.join("models", file_name + ".pt"))
        else:
            T.save(self.qNetwork.state_dict(), os.path.join("models", self.fileName))

    def load_net(self):
        """
        Loads the Q-Network from a file.
        :return: None
        """
        self.qNetwork.load_state_dict(T.load(os.path.join("models", self.fileName)))
