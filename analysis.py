#!/usr/bin/env python

import numpy as np
import re
import matplotlib.pyplot as plt

BIN_LEN = 20

epoch_scores = []
episode_scores = []



with open("logs_20180110_145657.txt") as f:
    for line in f:
        if "Reward" in line:
            matched = re.match(r"Epoch: ([0-9]+)\sReward: ([0-9]+)\sAVG Q-Max: ([0-9]+)", line)
            epoch = matched.group(1)
            reward = matched.group(2)
            avg_qmax = matched.group(3)
            epoch_scores += [[epoch, reward, avg_qmax]]
        elif "Score" in line:
            matched = re.match(r"Score: ([0-9]+)\sEpoch: ([0-9]+)\s", line)
            score = matched.group(1)
            epoch = matched.group(2)
            episode_scores += [[epoch, score]]

epoch_scores = np.array(epoch_scores).astype("float")
episode_scores = np.array(episode_scores).astype("float")

print(episode_scores[:, 1].shape)

plt.plot(epoch_scores[:, 0], epoch_scores[:, 1], "r^", label="Reward")
# plt.plot(epoch_scores[:, 0], epoch_scores[:, 2], "b--", label="AVG Q-Max")
coef = np.polyfit(epoch_scores[:, 0], epoch_scores[:, 1], 1)
fit_fn = np.poly1d(coef)
plt.plot(epoch_scores[:, 0], fit_fn(epoch_scores[:, 0]), "b--")
plt.legend()
plt.show()


i = np.arange(len(episode_scores[:, 0]))
coef = np.polyfit(i, episode_scores[:, 1], 1)
fit_fn = np.poly1d(coef)
plt.plot(episode_scores[:, 1], "r+", label="Reward")
plt.plot(i, fit_fn(i), "b--", label="Reward")
plt.show()


bins = len(episode_scores) // BIN_LEN
scores = episode_scores[:bins * BIN_LEN, 1].reshape(-1, BIN_LEN)

i = np.arange(bins)

means = np.mean(scores, axis=1)
stds = np.std(scores, axis=1)

coef = np.polyfit(i, means, 1)
print(coef)
means_lin = np.poly1d(coef)

coef = np.polyfit(i, stds, 1)
print(coef)
stds_lin = np.poly1d(coef)

plt.bar(i, means, color="red", label="mean")
plt.bar(i, stds, color="blue", label="std")
plt.plot(i, means_lin(i), "r--", linewidth=3)
plt.plot(i, stds_lin(i), "b--", linewidth=3)
plt.legend()
plt.show()
