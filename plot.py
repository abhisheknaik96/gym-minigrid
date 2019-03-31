import matplotlib.pyplot as plt
import numpy as np

window_length = 200

def get_vector(size):
    return np.concatenate((np.zeros(size-1), np.ones(1)))

def get_diff_sarsa_data():

    data = np.load('results/rewards_diff_sarsa.npy').item()
    step_sizes = data["step_sizes"]
    betas = data["betas"]
    all_rewards = data["all_rewards"]
    num_runs = data["num_runs"]

    data = np.convolve(np.ones(window_length),
                       np.mean(all_rewards[0][0], axis=0) / window_length,
                       mode='valid')

    return data

def get_sarsa_data():

    data = np.load('results/rewards_sarsa.npy').item()
    step_sizes = data["step_sizes"]
    all_rewards = data["all_rewards"]
    num_runs = data["num_runs"]
    num_eps = all_rewards.shape[2]

    # eq_data = np.zeros((len(step_sizes), num_runs, run_length))
    eq_data = [[[] for _ in range(num_runs)] for _ in range(len(step_sizes))]

    for i,_ in enumerate(step_sizes):

        run_length = min([np.int(np.sum(all_rewards[i][run])) for run in range(num_runs)])
        # print(run_length)

        for run in range(num_runs):
            tmp = get_vector(all_rewards[i][run][0])
            for eps in range(1,num_eps):
                tmp = np.concatenate((tmp, get_vector(all_rewards[0][run][eps])))
            eq_data[i][run] = tmp[:run_length]

    # data = np.convolve(np.ones(window_length),
    #                    np.mean(eq_data[0], axis=0)/window_length,
    #                    mode='valid')

    return eq_data, step_sizes
    # plt.plot(data)
    #
    # plt.title('Average reward over time (Sarsa)')
    # plt.legend(loc='upper right')
    # plt.show()

def plot_learning_curve_per_run(window_length=1000):

    for run in range(num_runs):
        data = np.convolve(np.ones(window_length),
                           np.mean(all_rewards[0][1], axis=0)/window_length,
                           mode='valid')
        plt.plot(data, label='Run ' + str(run))

    plt.title('Average reward over time')
    plt.legend(loc='upper right')
    plt.show()


def plot_learning_curve(step_size_idx, beta_idx, window_length=1000):

    data = np.convolve(np.ones(window_length),
                       np.mean(all_rewards[step_size_idx][beta_idx], axis=0)/window_length,
                       mode='valid')
    plt.plot(data)

    plt.title('Average reward over time')
    plt.legend(loc='upper right')
    plt.show()


def print_sweep_metrics():
    for step_idx, step_size in enumerate(step_sizes):
        for beta_idx, beta in enumerate(betas):
            print('Alpha=%f, Beta=%f, AvgReward=%f' % (step_size, beta,
                                                       np.mean(np.mean(all_rewards[step_idx][beta_idx]))))


def make_action_values_heatmap(action_values, avg_value=0.0):

    data = np.max(action_values, axis=1)
    data = np.rot90(np.reshape(data, (8,8)))
    print(data)
    data -= avg_value
    print(avg_value)
    print(data)
    # plt.imshow(data, cmap='hot', interpolation='nearest')
    heatmap = plt.pcolor(data)
    plt.colorbar(heatmap)
    # plt.title(r'Differential Sarsa, $\epsilon=0.2$')
    plt.title(r'Differential Sarsa, initialized to 2.0')

    plt.show()


def plot_learned_avg_values(avg_values, kappas):
    data = np.mean(avg_values[0][0], axis=1)
    plt.plot(kappas, data, marker='o')
    plt.xlabel(r'Step-size $\kappa$')
    plt.ylabel('Learned Average Value')
    plt.show()

# plot_learning_curve_per_run(2000)
# print_sweep_metrics()
# plot_learning_curve(0, 0, 2000)

# data_sarsa, step_sizes_sarsa = get_sarsa_data()
# data_diff_sarsa = get_diff_sarsa_data()

# for i in range(len(data_sarsa)):
#     data = np.convolve(np.ones(window_length),
#                        np.mean(data_sarsa[i], axis=0) / window_length,
#                        mode='valid')
#     # print(data[-1], step_sizes_sarsa[i])
#     plt.plot(data, label=step_sizes_sarsa[i])

# data_sarsa = np.convolve(np.ones(window_length),
#                          np.mean(data_sarsa[0], axis=0) / window_length,
#                          mode='valid')
#
# plt.plot(data_sarsa, label='Sarsa')
# plt.plot(data_diff_sarsa[:20000], label='Diff Sarsa')
#
# print(data_sarsa[15000], data_diff_sarsa[15000])
#
# # plt.title('Average reward over time for Sarsa with various step-sizes')
# plt.title('Average reward over time')
# plt.xlabel('Timesteps')
# plt.ylabel('Avg. reward')
# plt.legend(loc='center right')
# plt.show()

# data = np.load('results/rewards_sarsa.npy').item()
data = np.load('results/rewards_diff_sarsa.npy').item()
# action_values = data['action_values']
# make_action_values_heatmap(action_values, data['average_value'])

avg_values = data['avg_values']
kappas = data['kappas']
plot_learned_avg_values(avg_values, kappas)

# plt.plot(data["average_rewards"])
# plt.title('Learned average reward with time, initialized to 5.0')
# plt.xlabel('Timesteps')
# plt.ylabel(r'$\bar{R}$', rotation=0)
# plt.show()