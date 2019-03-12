import numpy as np
from optparse import OptionParser
import gym
import gym_minigrid
import time
from sarsa_agent import SarsaAgent, DifferentialSarsaAgent

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
        default=1000
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
        default=100000
    )
    (options, args) = parser.parse_args()

    all_rewards = []

    for run in range(options.num_runs):

        seed = options.seed + run
        # Load the gym environment
        env = gym.make(options.env_name)
        env.seed(seed)

        agent = DifferentialSarsaAgent()
        agent_info = {"num_states"  : 64,
                      "num_actions" : 4,
                      "epsilon"     : 0.2,
                      "step-size"   : 0.1,
                      "beta"        : 0.1,
                      "random_seed" : seed}
        agent.agent_init(agent_info=agent_info)

        obs = env.reset()
        action = agent.agent_start(obs)

        # sum_rewards = 0.0
        rewards = []

        for _ in range(options.run_length):

            obs, reward, done, info = env.step(action)
            action = agent.agent_step(reward, obs)

            # sum_rewards += reward
            rewards.append(reward)

            ### For visualization
            # renderer = env.render()
            # time.sleep(options.pause)

            # if renderer.window is None:
            #     break

        all_rewards.append(rewards)
        print('Run=%s, AvgReward=%.4f' % (run+1, sum(rewards)/options.run_length))


        # # Visualization after training
        # obs = env.reset()
        # action = agent.agent_start(obs)
        #
        # for _ in range(100):
        #
        #     obs, reward, done, info = env.step(action)
        #     action = agent.choose_action(obs)
        #     renderer = env.render()
        #     time.sleep(options.pause)
        #
        #     if renderer.window is None:
        #         break

    np.save('results/rewards', all_rewards)

if __name__ == "__main__":
    main()
