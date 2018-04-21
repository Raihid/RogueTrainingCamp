#!/usr/bin/python3

import datetime
import tensorflow as tf
import numpy as np
import rtc
import random
import time
import os

ALPHA = 1e-4
GAMMA = 0.99
INITIAL_EPSILON = 0.99
MINIMUM_EPSILON = 0.075

EXPERIMENT = 10000
EXPLORATION = 1000000
# EXPLORATION = 200000
MEMORY_SIZE = 120000
MINIBATCH_SIZE = 32

DELTA_EPS = (INITIAL_EPSILON - MINIMUM_EPSILON) / EXPLORATION

ONEHOT_LEN = len(rtc.Wrapper.CHAR_BINS)

def current_milli_time():
    return int(round(time.time() * 1000))


def conv_relu(x, W, stride, b):
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    relu = tf.nn.relu(conv + b)
    return relu


def generate_node(weight_shape):
        weight = tf.truncated_normal(shape=weight_shape, stddev=0.01)
        bias = tf.constant(0.01, shape=[weight_shape[-1]])
        return tf.Variable(weight), tf.Variable(bias)


class DeepLearningTrainer:

    def __init__(self):
        self.env = rtc.Wrapper()
        self.batch = []
        self.session = tf.InteractiveSession()
        self.epsilon = INITIAL_EPSILON
        self.epochs = 0
        self.holders = {"actions": tf.placeholder("float",
                                                  [None, self.env.ACTIONS]),
                        "target_fun":  tf.placeholder("float", [None]),
                        "obs": tf.placeholder("float", [None, 24, 80, ONEHOT_LEN])}
        self.holders["results"] = self.build_network("q_network")
        self.holders["target_results"] = self.build_network("target_q_network")
        self.holders["training_step"] = self.build_optimizer()
        self.update_target_op = self.build_target_updater()
        log_dir = "logs"
        log_name =  datetime.datetime.now().strftime("logs_%Y%m%d_%H:%M.txt")
        self.log_file = open(log_dir + "/" + log_name, "w")

    def build_network(self, name):
        with tf.variable_scope(name):

            with tf.variable_scope("first_tower"):
                weights = [None] * 3
                bias = [None] * 3
                layer = [None] * 2
                (weights[0], bias[0]) = generate_node([8, 8, ONEHOT_LEN, 16])
                (weights[1], bias[1]) = generate_node([4, 4, 16, 16])
                (weights[2], bias[2]) = generate_node([2, 2, 16, 16])

                layer[0] = conv_relu(self.holders["obs"], weights[0], 4, bias[0])
                layer[1] = conv_relu(layer[0], weights[1], 2, bias[1])
                first_tower_output = conv_relu(layer[1], weights[2], 1, bias[2])

            with tf.variable_scope("second_tower"):
                weights = [None] * 2
                bias = [None] * 2
                layer = [None] * 1
                (weights[0], bias[0]) = generate_node([6, 6, ONEHOT_LEN, 16])
                (weights[1], bias[1]) = generate_node([2, 2, 16, 16])

                layer[0] = conv_relu(self.holders["obs"], weights[0], 2, bias[0])
                second_tower_output = conv_relu(layer[0], weights[1], 1, bias[1])


            with tf.variable_scope("third_tower"):
                weights = [None] * 3
                bias = [None] * 3
                layer = [None] * 2
                (weights[0], bias[0]) = generate_node([4, 4, ONEHOT_LEN, 8])
                (weights[1], bias[1]) = generate_node([3, 3, 8, 8])
                (weights[2], bias[2]) = generate_node([2, 2, 8, 8])

                layer[0] = conv_relu(self.holders["obs"], weights[0], 2, bias[0])
                layer[1] = conv_relu(layer[0], weights[1], 1, bias[1])
                third_tower_output = conv_relu(layer[1], weights[2], 1, bias[2])

            weights = [None] * 3
            bias = [None] * 3
            layer = [None] * 2

            flat_first_tower = tf.reshape(first_tower_output, [-1, 3 * 10 * 16])
            flat_second_tower = tf.reshape(second_tower_output, [-1, 12 * 40 * 16])
            flat_third_tower = tf.reshape(third_tower_output, [-1, 12 * 40 * 8])

            flat_towers = tf.concat([flat_first_tower,
                                     flat_second_tower,
                                     flat_third_tower], axis=1)

            (weights[0], bias[0]) = generate_node([12000, 512])
            (weights[1], bias[1]) = generate_node([512, self.env.ACTIONS])


            layer[0] = tf.matmul(flat_towers, weights[0])
            layer[1] = tf.nn.relu(layer[0] + bias[0])

        return tf.matmul(layer[1], weights[1]) + bias[1]

    def build_optimizer(self):
        with tf.variable_scope("optimizer"):
            product = tf.multiply(self.holders["results"], self.holders["actions"])
            results_v = tf.reduce_sum(product, reduction_indices=1)
            square_error = tf.square(self.holders["target_fun"] - results_v)
            cost = tf.reduce_mean(square_error)
            training_step = tf.train.AdamOptimizer(ALPHA).minimize(cost)
        return training_step

    def initial_obs(self):
        blank = np.zeros((24, 80, ONEHOT_LEN))
        action = 0
        obs, reward, terminal, info = self.env.step(action)
        return obs, action, reward, terminal

    def load_weights(self):
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Weights loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Weights not found")
        return saver

    def choose_action(self, results_v):
        if random.random() < self.epsilon:
            return random.randrange(self.env.ACTIONS), True
        else:
            return np.argmax(results_v), False


    def feed_forward(self, obs):
        input_dict = {self.holders["obs"]: [obs]}
        results_v = self.holders["results"].eval(feed_dict=input_dict)[0]

        chosen_action, randomized = self.choose_action(results_v)
        print(results_v, "RAND" if randomized else "")

        action_v = [0] * self.env.ACTIONS
        action_v[chosen_action] = 1

        self.epsilon = max(self.epsilon - DELTA_EPS, MINIMUM_EPSILON)

        new_obs, reward, terminal, info = self.env.step(chosen_action)
        return new_obs, action_v, reward, terminal, max(results_v)

    def learn_minibatch(self):
        sample = random.sample(self.batch, MINIBATCH_SIZE)
        feed_dict = {self.holders["obs"]: [member[3] for member in sample]}

        target_readout, readout = self.session.run(
                (self.holders["target_results"],
                 self.holders["results"]),
                feed_dict=feed_dict)
        y = [None] * MINIBATCH_SIZE
        for idx, member in enumerate(sample):
            y[idx] = member[2]
            if member[4] is False:
                y[idx] += GAMMA * readout[idx][np.argmax(target_readout[idx])]

        self.holders["training_step"].run(feed_dict={
            self.holders["target_fun"]: y,
            self.holders["actions"]: [member[1] for member in sample],
            self.holders["obs"]: [member[0] for member in sample]})

    def experiment_phase(self, obs):
        while self.epochs < EXPERIMENT:
            chosen_action = random.randrange(self.env.ACTIONS)
            action_v = [0] * self.env.ACTIONS
            action_v[chosen_action] = 1
            new_obs, reward, terminal, info = self.env.step(chosen_action)
            self.batch.append((obs, action_v, reward, new_obs, terminal))

            obs = new_obs
            self.epochs += 1
        return obs

    def build_target_updater(self):
        q_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope="q_network")
        target_q_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope="target_q_network")

        target_updates = []
        for var, target_var in zip(sorted(q_vars, key=lambda v: v.name),
                                   sorted(target_q_vars, key=lambda v: v.name)):
            target_updates += [target_var.assign(var)]
        return tf.group(*target_updates)


    def train_network(self):
        savepoint = self.load_weights()
        obs, action_v, reward, terminal = self.initial_obs()
        obs = self.experiment_phase(obs)

        reward_sum = reward
        q_sum = 0
        while True:
            millis = current_milli_time()
            new_obs, action_v, reward, terminal, q = self.feed_forward(obs)
            self.batch.append((obs, action_v, reward, new_obs, terminal))

            if(len(self.batch) > MEMORY_SIZE):
                self.batch.pop(0)

            reward_sum += reward
            q_sum += q

            self.learn_minibatch()

            if terminal is True:
                score = self.env.score_for_logs()
                self.log_file.write("Score: {}\tEpoch: {}\n"
                                    .format(score, self.epochs))
                self.log_file.flush()

            if self.epochs % 5000 == 0:
                self.session.run(self.update_target_op)

            if self.epochs % 5000 == 0:
                avg_q = q_sum / 5000.0
                self.log_file.write("Epoch: {}\tReward: {}\tAVG Q-Max: {}\n"
                                    .format(self.epochs, reward_sum, avg_q))
                self.log_file.flush()
                q_sum = 0
                reward_sum = 0

            if self.epochs % 10000 == 0:
                savepoint.save(self.session,
                               'saved_networks/{}-dqn'.format(self.env.NAME),
                               global_step=self.epochs)

            print("Epoch: {}\tR: {}\tA: {}\t Eps: {}\tT: {}".
                  format(self.epochs, reward, np.argmax(action_v),
                         self.epsilon, current_milli_time() - millis))
            obs = new_obs
            self.epochs += 1

        self.log_file.close()

    def run(self):
        self.session.run(tf.initialize_all_variables())
        # obs, _, _, _ = self.initial_obs()
        # input_dict = {self.holders["obs"]: [obs]}
        # towers = self.session.run(self.towers, feed_dict=input_dict)
        self.train_network()

if __name__ == "__main__":
    dlt = DeepLearningTrainer()
    dlt.run()
