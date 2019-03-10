import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid
from sarsa_agent import SarsaAgent

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-8x8-cont-v0'
    )
    parser.add_option(
        "--seed",
        type=int,
        help="random seed (default: 1)",
        default=1
    )
    parser.add_option(
        "--pause",
        type=float,
        default=0.1,
        help="pause duration between two consequent actions of the agent"
    )
    parser.add_option(
        "--num_runs",
        type=int,
        help="number of runs of experiment to average over",
        default=1
    )
    parser.add_option(
        "--run_length",
        type=int,
        help="number of timesteps of a single run",
        default=10000
    )
    (options, args) = parser.parse_args()



    for run in range(options.num_runs):

        seed = options.seed + run
        # Load the gym environment
        env = gym.make(options.env_name)
        env.seed(seed)

        agent = SarsaAgent()
        agent_info = {"num_states" : 64, "num_actions" : 4, "epsilon" : 0.2, "random_seed" : seed}
        agent.agent_init(agent_info=agent_info)

        obs = env.reset()
        action = agent.agent_start(obs)

        sum_rewards = 0.0

        for _ in range(options.run_length):

            obs, reward, done, info = env.step(action)
            action = agent.agent_step(reward, obs)

            sum_rewards += reward

            # renderer = env.render()
            # time.sleep(options.pause)

            # if renderer.window is None:
            #     break

        print('Run=%s, AvgReward=%.4f' % (run+1, sum_rewards/options.run_length))


if __name__ == "__main__":
    main()
