import numpy as np
import random
import logging

from operator import attrgetter
from replay_buffer import ReplayBuffer

from banana_model import BananaNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import banana_config as bc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_format = "%(levelname)s %(asctime)s %(message)s"
logging.basicConfig(
    filename="banana_logger.log", level=logging.INFO, format=log_format, filemode="w"
)
logger = logging.getLogger()


def copy_parameters(main_network, target_network):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_params, main_params in zip(
        target_network.parameters(), main_network.parameters()
    ):
        target_params.data.copy_(
            bc.tau * main_params.data + (1.0 - bc.tau) * target_params.data
        )


class AgentDQN:
    """Deep-Q learning agent."""

    def __init__(self, state_size, action_size, sampling_method, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        self.qnetwork_main = BananaNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = BananaNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_main.parameters(), lr=bc.lr)

        # Replay memory
        self.memory = ReplayBuffer(bc.replay_buffer_size, bc.batch_size, seed)
        self.sampling_method = sampling_method

        # Initialize time step
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        priority = 1
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % bc.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > bc.batch_size:
                experiences, weights = self.memory.sample(method=self.sampling_method)
                loss = self.learn(experiences, weights)
                priority = torch.abs(loss).clone().detach() + 0.1

        # Save experience in replay memory
        if self.sampling_method == "random":
            self.memory.add(state, action, reward, next_state, done)
        elif self.sampling_method == "priority":
            self.memory.add(state, action, reward, next_state, done, priority)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_main.eval()
        with torch.no_grad():
            action_values = self.qnetwork_main(state)
        self.qnetwork_main.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, weights=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(1)

        # target values from the target network
        targets = torch.max(self.qnetwork_target.forward(next_states), 1)[0]
        targets = torch.unsqueeze(targets, 1)
        targets = targets * bc.gamma * (1 - dones) + rewards

        # predictions by the main network
        preds = self.qnetwork_main.forward(states)
        preds = torch.gather(preds, 1, actions)

        loss = F.mse_loss(preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        copy_parameters(self.qnetwork_main, self.qnetwork_target)

        return loss


class AgentDDQN(AgentDQN):
    """Double Deep-Q learning agent."""

    def learn(self, experiences, weights):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        # states, actions, rewards, next_states, dones, _ = experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(1)

        # pick the best action for the next state using the main network
        qvalues = self.qnetwork_main.forward(next_states)
        best_actions = torch.max(qvalues, 1)[1]
        best_actions = torch.unsqueeze(best_actions, 1)

        # calculate the Q values from target network
        target_qs = self.qnetwork_target.forward(next_states)
        targets = torch.gather(target_qs, 1, best_actions)
        targets = targets * bc.gamma * (1 - dones) + rewards

        # predictions by the main network
        preds = self.qnetwork_main.forward(states)
        preds = torch.gather(preds, 1, actions)

        loss = torch.sum(weights * (preds - targets) ** 2)
        # print("loss=", loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        copy_parameters(self.qnetwork_main, self.qnetwork_target)

        return torch.abs(loss)
