#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import rtc
import random
import time

ALPHA = 1e-4
GAMMA = 0.99
INITIAL_EPSILON = 0.99
MINIMUM_EPSILON = 0.075

EXPERIMENT = 10000
EXPLORATION = 800000
MEMORY_SIZE = 40000
MINIBATCH_SIZE = 32
DELTA_EPS = (INITIAL_EPSILON - MINIMUM_EPSILON) / EXPLORATION


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
        self.game = rtc.Wrapper()
        self.batch = []
        self.session = tf.InteractiveSession()
        self.epsilon = INITIAL_EPSILON
        self.epochs = 0
        self.holders = {"actions": tf.placeholder("float",
                                                  [None, self.game.ACTIONS]),
                        "target_fun":  tf.placeholder("float", [None]),
                        "state": tf.placeholder("float", [None, 80, 80, 4])}
        self.holders["results"] = self.build_network()
        self.holders["training_step"] = self.build_optimizer()
        self.log_file = open("logs.txt", "w")

    def build_network(self):
        weights = [None] * 5
        bias = [None] * 5
        layer = [None] * 4

        (weights[0], bias[0]) = generate_node([8, 8, 4, 32])
        (weights[1], bias[1]) = generate_node([4, 4, 32, 64])
        (weights[2], bias[2]) = generate_node([3, 3, 64, 64])
        (weights[3], bias[3]) = generate_node([6400, 512])
        (weights[4], bias[4]) = generate_node([512, self.game.ACTIONS])

        layer[0] = conv_relu(self.holders["state"], weights[0], 4, bias[0])
        layer[1] = conv_relu(layer[0], weights[1], 2, bias[1])
        layer[2] = conv_relu(layer[1], weights[2], 1, bias[2])
        full_layer = tf.matmul(tf.reshape(layer[2], [-1, 6400]), weights[3])
        layer[3] = tf.nn.relu(full_layer + bias[3])
        return tf.matmul(layer[3], weights[4]) + bias[4]

    def build_optimizer(self):
        product = tf.multiply(self.holders["results"], self.holders["actions"])
        results_v = tf.reduce_sum(product, reduction_indices=1)
        square_error = tf.square(self.holders["target_fun"] - results_v)
        cost = tf.reduce_mean(square_error)
        training_step = tf.train.AdamOptimizer(ALPHA).minimize(cost)
        return training_step

    def initial_state(self):
        blank = np.zeros((80, 80))
        state = np.stack((blank, blank, blank, blank), axis=2)
        action = 0
        reward, terminal = self.game.next_tick(action)
        return state, action, reward, terminal

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
            return random.randrange(self.game.ACTIONS), True
        else:
            return np.argmax(results_v), False

    def process_frame(self, state):
        pixels = self.game.get_frame()
        # squashed = np.empty([80, 80])
        # squashed = pixels.reshape(80, 2, 80, 2).sum(axis=1).sum(axis=2)
        processed_frame = pixels.reshape(80, 80, 1)

        return np.append(processed_frame, state[:, :, 0:3], axis=2)

    def feed_forward(self, state):
        input_dict = {self.holders["state"]: [state]}
        results_v = self.holders["results"].eval(feed_dict=input_dict)[0]

        chosen_action, randomized = self.choose_action(results_v)
        print(results_v, "RAND" if randomized else "")

        action_v = [0] * self.game.ACTIONS
        action_v[chosen_action] = 1

        self.epsilon = max(self.epsilon - DELTA_EPS, MINIMUM_EPSILON)

        reward, terminal = self.game.next_tick(chosen_action)
        new_state = self.process_frame(state)
        return new_state, action_v, reward, terminal, max(results_v)

    def learn_minibatch(self):
        sample = random.sample(self.batch, MINIBATCH_SIZE)
        feed_dict = {self.holders["state"]: [member[3] for member in sample]}

        readout = self.holders["results"].eval(feed_dict=feed_dict)
        y = [None] * MINIBATCH_SIZE
        for idx, member in enumerate(sample):
            y[idx] = member[2]
            if member[4] is False:
                y[idx] += GAMMA * np.max(readout[idx])

        self.holders["training_step"].run(feed_dict={
            self.holders["target_fun"]: y,
            self.holders["actions"]: [member[1] for member in sample],
            self.holders["state"]: [member[0] for member in sample]})

    def experiment_phase(self, state):
        while self.epochs < EXPERIMENT:
            chosen_action = random.randrange(self.game.ACTIONS)
            action_v = [0] * self.game.ACTIONS
            action_v[chosen_action] = 1
            reward, terminal = self.game.next_tick(chosen_action)
            new_state = self.process_frame(state)
            self.batch.append((state, action_v, reward, new_state, terminal))

            state = new_state
            self.epochs += 1
        return state

    def train_network(self):
        savepoint = self.load_weights()
        state, action_v, reward, terminal = self.initial_state()
        state = self.experiment_phase(state)

        reward_sum = reward
        q_sum = 0
        while True:
            millis = current_milli_time()
            new_state, action_v, reward, terminal, q = self.feed_forward(state)
            self.batch.append((state, action_v, reward, new_state, terminal))
            if(len(self.batch) > MEMORY_SIZE):
                self.batch.pop(0)

            reward_sum += reward
            q_sum += q

            self.learn_minibatch()

            if terminal is True:
                score = self.game.score_for_logs()
                self.log_file.write("Score: {}\tEpoch: {}\n"
                                    .format(score, self.epochs))
                self.log_file.flush()

            if self.epochs % 5000 == 0:
                avg_q = q_sum / 5000.0
                self.log_file.write("Epoch: {}\tReward: {}\tAVG Q-Max: {}\n"
                                    .format(self.epochs, reward_sum, avg_q))
                self.log_file.flush()
                q_sum = 0
                reward_sum = 0

            if self.epochs % 10000 == 0:
                savepoint.save(self.session,
                               'saved_networks/{}-dqn'.format(self.game.NAME),
                               global_step=self.epochs)

            print("Epoch: {}\tR: {}\tA: {}\t Eps: {}\tT: {}".
                  format(self.epochs, reward, np.argmax(action_v),
                         self.epsilon, current_milli_time() - millis))
            state = new_state
            self.epochs += 1

        self.log_file.close()

    def run(self):
        self.session.run(tf.initialize_all_variables())
        self.train_network()


if __name__ == "__main__":
    dlt = DeepLearningTrainer()
    dlt.run()
