import matplotlib.pyplot as plt
import numpy as np

data = np.load('results/rewards.npy').item()
step_sizes = data["step_sizes"]
betas = data["betas"]
all_rewards = data["all_rewards"]
num_runs = data["num_runs"]

def plot_learning_curve(window_length=1000):

    for run in range(num_runs):
        data = np.convolve(np.ones(window_length),
                           np.mean(all_rewards[0][0], axis=0)/window_length,
                           mode='valid')
        plt.plot(data, label='Run ' + str(run))

    plt.title('Average reward over time')
    plt.legend(loc='upper right')
    plt.show()

# plot_learning_curve(2000)
for step_idx, step_size in enumerate(step_sizes):
    for beta_idx, beta in enumerate(betas):
        print('Alpha=%f, Beta=%f, AvgReward=%f' % (step_size, beta, np.mean(np.mean(all_rewards[step_idx][beta_idx]))))