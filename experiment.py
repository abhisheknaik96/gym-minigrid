import numpy as np
from optparse import OptionParser
import gym
import gym_minigrid
import sys, time
from tqdm import tqdm
from sarsa_agent import SarsaAgent, DifferentialSarsaAgent

def visualize_after_training(env, agent, pause_length):

    obs = env.reset()
    action = agent.agent_start(obs)

    for _ in range(100):

        obs, reward, done, info = env.step(action)
        action = agent.choose_action(obs)
        renderer = env.render()
        time.sleep(pause_length)

        if renderer.window is None:
            break

def run_sarsa():

    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-8x8-v0'
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
        "--num_eps",  # for usual (non-differential) Sarsa
        type=int,
        help="number of episodes in a single run",
        default=1500
    )
    (options, args) = parser.parse_args()

    # step_sizes = [1e-3, 1e-2, 1e-1, 1.0]
    step_sizes = [1e-1]

    all_rewards = np.ndarray((len(step_sizes), options.num_runs, options.num_eps), dtype=np.int)
    log_data = {"step_sizes": step_sizes, "num_runs": options.num_runs, "betas" : []}

    for step_idx, step_size in enumerate(step_sizes):

        tqdm.write('Sarsa : Alpha=%f' % (step_size))

        for run in tqdm(range(options.num_runs), file=sys.stdout):

            seed = options.seed + run
            # Load the gym environment
            env = gym.make(options.env_name)
            env.seed(seed)

            agent = SarsaAgent()
            agent_info = {"num_states": 64,
                          "num_actions": 4,
                          "epsilon": 0.2,
                          "step-size": step_size,
                          "random_seed": seed}
            agent.agent_init(agent_info=agent_info)

            # rewards = []

            for eps in range(options.num_eps):

                sum_rewards = 0.0
                num_steps = 0
                done = 0
                obs = env.reset()
                action = agent.agent_start(obs)

                while done==0:

                    obs, reward, done, info = env.step(action)
                    if done!=1:
                        action = agent.agent_step(reward, obs)
                    else:
                        agent.agent_end(reward)
                    # no update when agent time-outs (done=2)

                    sum_rewards += reward
                    num_steps += 1

                    ### For visualization
                    # renderer = env.render()
                    # time.sleep(options.pause)

                    # if renderer.window is None:
                    #     break

                all_rewards[step_idx][run][eps] = num_steps

            log_data["action_values"] = agent.q_values
            ### Visualization after training
            # visualize_after_training(env, agent, options.pause)

        # all_rewards.append(rewards)
        tqdm.write('AvgReward_total\t\t= %f' % (1.0/(np.mean(np.mean(all_rewards[step_idx])))))

    log_data["all_rewards"] = all_rewards
    np.save('results/rewards_sarsa', log_data)


def run_diff_sarsa():
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
        default=20000
    )
    (options, args) = parser.parse_args()

    # step_sizes = [1e-3, 1e-2, 1e-1, 1.0]
    # betas = [1e-3, 1e-2, 1e-1, 1.0]
    step_sizes = [1e-1]
    betas = [1e-2]
    # kappas = [1e-3, 1e-2, 1e-1, 1.0]
    kappas = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    # kappas = [0.45, 0.55, 0.6, 0.65, 0.7]

    all_rewards = np.ndarray((len(step_sizes), len(betas), len(kappas), options.num_runs, options.run_length))
    avg_values = np.ndarray((len(step_sizes), len(betas), len(kappas), options.num_runs))
    log_data = {"step_sizes": step_sizes, "betas": betas, "kappas": kappas, "num_runs": options.num_runs}

    for step_idx, step_size in enumerate(step_sizes):

        for beta_idx, beta in enumerate(betas):

            for kappa_idx, kappa in enumerate(kappas):

                tqdm.write('DiffSarsa : Alpha=%f, Beta=%f, Kappa=%f' % (step_size, beta, kappa))

                for run in tqdm(range(options.num_runs), file=sys.stdout):

                    seed = options.seed + run
                    # Load the gym environment
                    env = gym.make(options.env_name)
                    env.seed(seed)

                    agent = DifferentialSarsaAgent()
                    agent_info = {"num_states": 64,
                                  "num_actions": 4,
                                  "epsilon": 0.2,
                                  "step-size": step_size,
                                  "beta": beta,
                                  "kappa": kappa,
                                  "random_seed": seed}
                    agent.agent_init(agent_info=agent_info)

                    obs = env.reset()
                    action = agent.agent_start(obs)

                    # sum_rewards = 0.0
                    # rewards = []
                    avg_rewards = []

                    for timestep in range(options.run_length):

                        obs, reward, done, info = env.step(action)
                        action = agent.agent_step(reward, obs)

                        # sum_rewards += reward
                        all_rewards[step_idx][beta_idx][kappa_idx][run][timestep] = reward


                        avg_rewards.append(agent.avg_reward)
                        ### For visualization
                        # renderer = env.render()
                        # time.sleep(options.pause)

                        # if renderer.window is None:
                        #     break

                    # Visualization after training
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

                    log_data["action_values"] = agent.q_values
                    log_data["average_rewards"] = avg_rewards
                    avg_values[step_idx][beta_idx][kappa_idx][run] = agent.avg_value
                    # tqdm.write("Avg. reward : %f" % agent.avg_reward)

                # all_rewards.append(rewards)

                tqdm.write('AvgReward_total\t\t= %f' % (np.mean(all_rewards[step_idx][beta_idx][kappa_idx])))
                tqdm.write('AvgReward_last1000\t= %f\n' % (np.mean(all_rewards[step_idx,beta_idx,kappa_idx,:,-1000:])))

    log_data["all_rewards"] = all_rewards
    log_data["avg_values"] = avg_values
    np.save('results/rewards_diff_sarsa', log_data)


def main():

    run_diff_sarsa()
    # run_sarsa()


if __name__ == "__main__":
    main()
