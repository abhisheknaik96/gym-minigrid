import numpy as np
from optparse import OptionParser
import gym
import gym_minigrid
import sys
from tqdm import tqdm
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
        default=10
    )
    parser.add_option(
        "--run_length",
        type=int,
        help="number of timesteps of a single run",
        default=50000
    )
    (options, args) = parser.parse_args()

    step_sizes = [1e-3, 1e-2, 1e-1, 1.0]
    betas = [1e-3, 1e-2, 1e-1, 1.0]
    # step_sizes = [1e-1]
    # betas = [1e-1]

    all_rewards = np.ndarray((len(step_sizes), len(betas), options.num_runs, options.run_length))
    log_data = {"step_sizes" : step_sizes, "betas" : betas, "num_runs" : options.num_runs}

    for step_idx, step_size in enumerate(step_sizes):

        for beta_idx, beta in enumerate(betas):

            tqdm.write('Alpha=%f, Beta=%f' % (step_size, beta))

            for run in tqdm(range(options.num_runs), file=sys.stdout):

                seed = options.seed + run
                # Load the gym environment
                env = gym.make(options.env_name)
                env.seed(seed)

                agent = DifferentialSarsaAgent()
                agent_info = {"num_states"  : 64,
                              "num_actions" : 4,
                              "epsilon"     : 0.2,
                              "step-size"   : step_size,
                              "beta"        : beta,
                              "random_seed" : seed}
                agent.agent_init(agent_info=agent_info)

                obs = env.reset()
                action = agent.agent_start(obs)

                # sum_rewards = 0.0
                # rewards = []

                for timestep in range(options.run_length):

                    obs, reward, done, info = env.step(action)
                    action = agent.agent_step(reward, obs)

                    # sum_rewards += reward
                    all_rewards[step_idx][beta_idx][run][timestep] = reward

                    ### For visualization
                    # renderer = env.render()
                    # time.sleep(options.pause)

                    # if renderer.window is None:
                    #     break


            # all_rewards.append(rewards)
            tqdm.write('AvgReward=%f\n' % (np.mean(np.mean(all_rewards[step_idx][beta_idx]))))


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

    log_data["all_rewards"] = all_rewards
    np.save('results/rewards', log_data)

if __name__ == "__main__":
    main()
