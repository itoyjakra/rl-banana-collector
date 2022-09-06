from agents import AgentDDQN, AgentDQN
from unityagents import UnityEnvironment
import banana_config as bc
import torch
import argparse


def go_bananas(agent, num_episodes=1):
    """Runs a trained agent in the banana collector env."""

    for episode in range(num_episodes):
        env_info = env_banana.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        sum_rewards = 0
        # for i in range(1000):
        while True:
            action = agent.act(state)
            env_info = env_banana.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            print(f"{action=}, {reward=}, {sum_rewards=}")
            sum_rewards += reward
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"{episode=}, total rewards = {sum_rewards}")
                break
    env_banana.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="agent_type", type=str, help="type of agent (DQN/DDQN)")
    parser.add_argument(dest="model_path", type=str, help="model path")
    parser.add_argument("-n", "--num_episodes", type=int, default=1)
    args = parser.parse_args()

    env_banana = UnityEnvironment(file_name=bc.unity_path)
    brain_name = env_banana.brain_names[0]
    brain = env_banana.brains[brain_name]

    if args.agent_type == "DQN":
        agent = AgentDQN(
            state_size=brain.vector_observation_space_size,
            action_size=brain.vector_action_space_size,
            sampling_method=None,
            seed=100,
        )
    elif args.agent_type == "DDQN":
        agent = AgentDDQN(
            state_size=brain.vector_observation_space_size,
            action_size=brain.vector_action_space_size,
            sampling_method=None,
            seed=100,
        )

    agent.qnetwork_main.load_state_dict(torch.load(args.model_path))
    go_bananas(agent=agent, num_episodes=args.num_episodes)
