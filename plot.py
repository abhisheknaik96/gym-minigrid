import matplotlib.pyplot as plt
import numpy as np

window_length = 200

data = np.load('results/rewards.npy')
data = np.convolve(np.ones(window_length), np.mean(data, axis=0)/window_length, mode='valid')

plt.plot(data)#, label='0 steps')
plt.title('Average reward over time')
plt.legend(loc='upper right')
plt.show()
