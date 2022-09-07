eps_start = 1.0  # starting value of epsilon (epsilon greedy policy)
eps_end = 0.01  # maximum value of epsilon
eps_decay = 0.995  # rate of epsilon decay
gamma = 0.99  # discount factor in Bellman equation
alpha = 0.9  # priority scale (0, 1), 0 => no priority
beta = 0.2  # importance    weight
tau = 1e-3  # for soft update of target network parameters
lr = 5e-4  # learning rate


replay_buffer_size = 100_000  # replay buffer size
batch_size = 64  # minibatch size for training
update_every = 16  # how often to update the target network
sampling_strategy = "random"  # how to access the replay memory

max_steps_in_episode = 1000  # the max number of steps to run
num_nodes = 256  # number of nodes used in the ANN layers 2 and 3

unity_path = "../Banana_Linux/Banana.x86_64"
unity_path_novis = "../Banana_Linux_NoVis/Banana.x86_64"
