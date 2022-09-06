# Banana Collector using a DDQN agent with Priority Buffer

insert video

## Environment

In this environment, an RL agent moves in a plane with random bananas scattered in a confined space. The goal is to collect as many yellow bananas as possible while avoiding the blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions to achive the above mentioned goal.

The agent can take one of the following four discrete actions in the environment:

* 0: move left
* 1: move right
* 2: move forward
* 3: move backward

There is a reward of +1 for grabbing a yellow banana and a reward of -1 for grabbing a blue banana.

The task is episodic in nature and the agent wins if it scores at least 13 points on the average over 100 consecutive episodes.


## Setup

To get started with training an RL agent in the Unity environment follow these steps:

1. Clone this repository
1. Download the appropriate Unity environment for your OS:

    * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
1. Create a new Conda environment and install the requirements
    ```
    conda create -n dqn python=3.8
    conda activate dqn
    pip install -r requirements.txt
    ```
## Training
To train a new agent, run the following from a terminal
```
python run_agent.py
```

## Observe a trained agent
```
python play_banana.py
```
