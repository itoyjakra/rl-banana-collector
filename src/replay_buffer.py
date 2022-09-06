from collections import deque
import random
import numpy as np
import torch
import banana_config as bc


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    # alpha = 0.5  # priority scale (0, 1), 0 => no priority

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        # self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.priority = deque(maxlen=buffer_size)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority=None):
        """Add a new experience to memory."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        if priority is not None:
            self.priority.append(priority)

    def sample(self, method, beta=0.8):
        """Randomly sample a batch of experiences from memory."""
        if method == "random":
            experiences = random.sample(self.memory, k=self.batch_size)
            weights = np.ones(len(self.memory))
        elif method == "priority":
            prios = np.array(self.priority) ** bc.alpha
            probs = prios / sum(prios)

            batch_indices = random.choices(
                range(len(self.priority)), k=self.batch_size, weights=probs
            )
            experiences = np.array(self.memory)[batch_indices]

            weights = (1 / len(self.memory) / np.array(probs)[batch_indices]) ** beta
            weights = weights / np.max(weights)

        return experiences, torch.tensor(weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
