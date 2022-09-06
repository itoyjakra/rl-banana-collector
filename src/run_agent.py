import logging
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from unityagents import UnityEnvironment
import numpy as np
from agents import AgentDQN, AgentDDQN
import banana_config as bc
import torch


def get_banana_env():
    """Returns the banana collector environment."""
    return UnityEnvironment(file_name=bc.unity_path_novis)


def run_episode(*, env, agent, epsilon):
    """Runs one episode for an episodic task and returns the score."""
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]

    score = 0
    # for _ in range(bc.max_time_in_episode):
    while True:
        action = agent.act(state, epsilon)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

    return score


def run_banana_collector(*, env, agent):
    """Runs an agent in the banana collector environment."""

    env_info = env.reset(train_mode=True)[brain_name]
    # state = env_info.vector_observations[0]

    scores = []
    scores_window = deque(maxlen=100)
    eps = bc.eps_start
    for time_step in range(bc.max_steps_in_episode):
        score = run_episode(env=env, agent=agent, epsilon=eps)
        scores_window.append(score)
        scores.append(score)
        eps = max(bc.eps_end, bc.eps_decay * eps)
        logger.info(
            f"Step = {time_step + 1}, Avg. Score = {np.mean(scores_window):.2f}"
        )

    env.close()

    return scores


def plot_scores(scores, window_size):
    """PLots the scores from running a numer of episodes."""
    scores = pd.Series(scores)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scores.plot(ax=ax)
    scores.rolling(window=window_size).mean().plot(ax=ax)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


if __name__ == "__main__":
    log_format = "%(levelname)s %(asctime)s %(message)s"
    logging.basicConfig(
        filename="banana_logger.log",
        level=logging.INFO,
        format=log_format,
        filemode="w",
    )
    logger = logging.getLogger()

    env_banana = get_banana_env()
    brain_name = env_banana.brain_names[0]
    brain = env_banana.brains[brain_name]
    agent_ddqn = AgentDDQN(
        state_size=brain.vector_observation_space_size,
        action_size=brain.vector_action_space_size,
        sampling_method=bc.sampling_strategy,
        seed=100,
    )

    scores = run_banana_collector(env=env_banana, agent=agent_ddqn)
    torch.save(agent_ddqn.qnetwork_main.state_dict(), "checkpoint.pth")
    scores = pd.Series(scores)
    scores.to_csv("run.csv", index=False)
    plot_scores(scores=scores, window_size=100)
