#!/usr/bin/env python

import numpy as np
import re
import matplotlib.pyplot as plt
import sys

BIN_LEN = 10


def get_scores(log_filename):
    step_scores = []
    episode_scores = []
    with open(log_filename) as f:
        for line in f:
            if "Reward" in line:
                matched = re.match((r"Epoch: ([0-9]+)\sReward: ([0-9]+)"
                                    r"\sAVG Q-Max: ([0-9]+)"), line)
                epoch = matched.group(1)
                reward = matched.group(2)
                avg_qmax = matched.group(3)
                step_scores += [[epoch, reward, avg_qmax]]
            elif "Score" in line:
                matched = re.match(r"Score: ([0-9]+)\sEpoch: ([0-9]+)\s", line)
                score = matched.group(1)
                epoch = matched.group(2)
                episode_scores += [[epoch, score]]

    step_scores = np.array(step_scores).astype("float")
    episode_scores = np.array(episode_scores).astype("float")

    return step_scores, episode_scores


def plot_rewards_in_time(step_scores):
    coef = np.polyfit(step_scores[:, 0], step_scores[:, 1], 1)
    fit_fn = np.poly1d(coef)
    print(("Linear regression for rewards in time "
           "{}*x + {}".format(coef[0], coef[1])))

    plt.plot(step_scores[:, 0], step_scores[:, 1], "ro", label="Reward")
    plt.plot(step_scores[:, 0], fit_fn(step_scores[:, 0]), "b--")

    plt.xlabel("Time (in steps)")
    plt.ylabel("Reward")
    plt.title("Reward in time")

    plt.legend()
    plt.show()


def plot_rewards_per_episode(episode_scores):
    i = np.arange(len(episode_scores[:, 0]))
    coef = np.polyfit(i, episode_scores[:, 1], 1)
    fit_fn = np.poly1d(coef)
    print(("Linear regression for rewards per episode "
           "{}*x + {}".format(coef[0], coef[1])))
    plt.plot(episode_scores[:, 1], "r+", label="Reward")

    plt.plot(i, fit_fn(i), "b--", label="Linear regression of reward")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per episode")

    plt.show()


def plot_bins(episode_scores):
    bins = len(episode_scores) // BIN_LEN
    scores = episode_scores[:bins * BIN_LEN, 1].reshape(-1, BIN_LEN)

    i = np.arange(bins)

    means = np.mean(scores, axis=1)
    means_coef = np.polyfit(i, means, 1)
    means_lin = np.poly1d(means_coef)
    print(("Linear regression for binned means: "
           "{}*x + {}".format(means_coef[0], means_coef[1])))

    stds = np.std(scores, axis=1)
    stds_coef = np.polyfit(i, stds, 1)
    stds_lin = np.poly1d(stds_coef)
    print(("Linear regression for binned standard deviations: "
           "{}*x + {}".format(stds_coef[0], stds_coef[1])))

    plt.bar(i, means, color="#ff3333", label="mean", alpha=0.7)
    plt.bar(i, stds, color="#3333ff", label="std", alpha=0.7)
    plt.plot(i, means_lin(i), "r--", linewidth=3)
    plt.plot(i, stds_lin(i), "b--", linewidth=3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("The first argument should be a filename!")
    log_filename = sys.argv[1]
    step_scores, episode_scores = get_scores(log_filename)
    plot_rewards_in_time(step_scores)
    plot_rewards_per_episode(episode_scores)
    plot_bins(episode_scores)
