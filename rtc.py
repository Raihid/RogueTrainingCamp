#!/usr/bin/env python3


# -*- coding: utf-8 -*-

import numpy as np
import os
import pty
import pyte
import random
import re
import select
import string
import sys
import time

from collections import deque
from gym import spaces

# TODO
# Przerobic to na API od OpenAI
# Sygnal po zakonczeniu tury od rogue
# Staty - analiza deskryptora, wysokie value stanu =>
# ktora czesc stanu odpowiada za to, saliency

# A3C, A2C
# Labirynty 3D, Unreal, RE with axilary tasks, mininagrody do chodzenia
# Logarytm z pol
# Advantage? Policy gradient, Sutton, Silvera
# Wyciac ostatnia linijke


executable = "/home/rahid/Programming/Repos/rogue5.2/rogue"
SCREEN_WIDTH = 80
SCREEN_HEIGHT = 24
CLIPPED_SCREEN_WIDTH = 80

SQUARE_SCREEN = False
ONE_HOT = True
SCREEN_DUMP_RATE = 10


class MultiWrapper():

    GAME_ACTIONS = [# " ",  # skip
                    "k",  # go_up
                    "j",  # go_down
                    "h",  # go_left
                    "l",  # go_right
                    # ">",  # descend
                    # "s",  # search
                    # "." search
                    ]
    ACTIONS = len(GAME_ACTIONS)
    NAME = "Rogue"
    CHAR_BINS = ["@", # Player char
                 string.ascii_letters, # Monsters
                 "-|",  # Walls
                 "+",  # Doors
                 ".#",  # Floors/corridors
                 "*?!%[]",  # Items
                 " ",  # Empty space
                 ""]  # Other

    def __init__(self):
        self.action_space = spaces.Discrete(len(self.GAME_ACTIONS))
        print(self.action_space.shape)
        self.observation_space = spaces.Box(0, 1,
                [SCREEN_HEIGHT, SCREEN_WIDTH, len(self.CHAR_BINS)])
        self.num_envs = 10
        self.envs = [Wrapper() for _ in range(self.num_envs)]

    def reset(self, restart=True):
        return np.array([env.reset(restart) for env in self.envs])

    def step(self, actions):
        multi_obs = []
        multi_rewards = []
        multi_dones = []

        for env, action in zip(self.envs(), actions):
            obs, reward, done, _ = env.step(action)
            multi_obs += [obs]
            multi_rewards += [reward]
            multi_dones += [done]

        return (np.array(multi_obs),
                np.array(multi_rewards),
                np.array(multi_dones),
                None)

    def close():
        pass


class Wrapper():
    GAME_ACTIONS = [# " ",  # skip
                    "k",  # go_up
                    "j",  # go_down
                    "h",  # go_left
                    "l",  # go_right
                    # ">",  # descend
                    # "s",  # search
                    # "." search
                    ]
    ACTIONS = len(GAME_ACTIONS)
    NAME = "Rogue"
    CHAR_BINS = ["@", # Player char
                 string.ascii_letters, # Monsters
                 "-|",  # Walls
                 "+",  # Doors
                 ".#",  # Floors/corridors
                 "*?!%[]",  # Items
                 " ",  # Empty space
                 ""]  # Other

    def __init__(self):
        self.state = []
        self.state_history = deque([], 10)
        self.scores = [0]
        self.action_history = deque(["EPOCH START"], 100)
        self.tick = 0
        self.last_death = 0
        self.process_screen = self.char_to_vec if ONE_HOT else ord
        self.action_space = spaces.Discrete(len(self.GAME_ACTIONS))
        self.observation_space = spaces.Box(0, 1,
                [SCREEN_HEIGHT, SCREEN_WIDTH, len(self.CHAR_BINS)])
        self.reset(restart=False)


    def _get_random_input(self):
        return random.choice(self.GAME_ACTIONS)

    def _dump_screen(self):
        for idx, line in enumerate(self.screen.display, 1):
                print("{0:2d} {1} =".format(idx, line))


    def score_for_logs(self):
        return self.scores[-2]

    def _scrap_screen(self):
        info_line = self.state[-1]

        if "Gold" not in info_line:  # there's probably a message there
            self._push_action("\r\r\n")  # push return to skip
            self._push_action(" ")  # more
            self._push_action(" ")  # more
            info_line = self.screen.display[-1]
            time.sleep(0.2)
            self._dump_screen()
            print(info_line)

        heuristics = {}
        heuristics["dungeon_level"] = int(
                re.search("Level:\s+(\d+)", info_line).group(1))
        heuristics["gold"] = int(re.search("Gold:\s+(\d+)", info_line).group(1))

        hp_captures = re.search("Hp:\s+(\d+)\((\d+)\)", info_line)
        heuristics["hp"] = int(hp_captures.group(1))
        heuristics["max_hp"] = int(hp_captures.group(2))

        exp_captures = re.search("Exp:\s+(\d+)/(\d+)", info_line)
        heuristics["exp_level"] = int(exp_captures.group(1))
        heuristics["exp_points"] = int(exp_captures.group(2))

        if len(self.state_history) >= 2:
            new_explored = sum(not c.isspace()
                               for line in self.state[1:-1] for c in line)
            old_explored = sum(not c.isspace()
                               for line in self.state_history[-2][1:-1] for c in line)
            heuristics["explored"] = new_explored - old_explored
        else:
            heuristics["explored"] = 0


        if len(self.state_history) >= 10:
            old_pos = self._get_player_pos(self.state_history[0])
            new_pos = self._get_player_pos()
            distance = abs(old_pos[0] - new_pos[0]) + abs(old_pos[1] - new_pos[1])
            heuristics["pos_diff"] = (distance - 4) * 0.2
        else:
            heuristics["pos_diff"] = 0

        heuristics["constant"] = -0.05

        return heuristics

    def _calculate_reward(self):
        try:
            heuristics = self._scrap_screen()
        except AttributeError:
            return self.scores[-1]
        return heuristics["explored"] + heuristics["pos_diff"] + heuristics["constant"]
        # + (heuristics["dungeon_level"] - 1) * 1000)

    def _push_action(self, input_):
        for input_char in input_:
            os.write(self.master_fd, bytearray(input_char, "ascii"))
            time.sleep(0.001)

    def _get_player_pos(self, state=None):
        if state is None:
            state = self.state
        for y, line in enumerate(state):
            x = line.find("@")
            if x != -1:
                return x, y

    def char_to_vec(self, char):
        if ord(char) == 0:
            return [0] * len(self.CHAR_BINS)

        for idx, char_bin in enumerate(self.CHAR_BINS):
            pos = char_bin.find(char)
            if pos != -1:
                bin_index = idx
                break
        else:
            bin_index = len(self.CHAR_BINS) - 1
            pos = 0

        one_hot = [0] * len(self.CHAR_BINS)
        one_hot[bin_index] = 1
        return one_hot

    def get_clipped_edges(self):
        if not SQUARE_SCREEN:
            return 0, SCREEN_WIDTH
        player_pos = self._get_player_pos()
        player_left_edge = player_pos[0] - CLIPPED_SCREEN_WIDTH // 2

        max_left_edge = SCREEN_WIDTH - CLIPPED_SCREEN_WIDTH
        min_left_edge = 0
        left_edge = sorted([min_left_edge, player_left_edge, max_left_edge])[1]
        right_edge = left_edge + CLIPPED_SCREEN_WIDTH

        return left_edge, right_edge

    def print_char_frame(self):
        left_edge, right_edge = self.get_clipped_edges()

        for line in self.state:
            print(line[left_edge:left_edge + CLIPPED_SCREEN_WIDTH])

    def get_frame(self):
        left_edge, right_edge = self.get_clipped_edges()

        processed_state = []
        for line in self.state:
            processed_state += [[self.process_screen(c) for c
                                 in line[left_edge:right_edge]]]

        padding = (CLIPPED_SCREEN_WIDTH - SCREEN_HEIGHT) if SQUARE_SCREEN else 0
        for _ in range(padding):
            null_element = self.process_screen('\0')
            processed_state += [[null_element] * CLIPPED_SCREEN_WIDTH]

        return np.array(processed_state)

    def run_in_loop(self):
        tick = 0
        time.sleep(0.2)

        while True:
            read_list, _wlist, _xlist = select.select(
                [self.master_fd], [], [], 0)

            if not read_list:
                # No more output, let's process results
                terminal = self._is_terminal()
                self._push_action(" ")
                if terminal:
                    self._start_game(restart=True)

                tick += 1
                self.state = self.screen.display
                self.scores += [self._calculate_reward()]
                self.render()
                game_input = self._get_input()
                self._push_action(game_input)
                self.action_history += [game_input]
            else:
                data = os.read(read_list[0], 1024)
                self.stream.feed(data.decode("ascii"))

    def _is_terminal(self):
        screen = "".join(line for line in self.screen.display)
        return ("REST" in screen and "IN" in screen and "PEACE" in screen)

    def reset(self, restart=True):
        if restart:
            self._push_action(" ")
            self._push_action("\r\n")
            self._push_action("^C")
            os.close(self.master_fd)
            os.wait()

        p_pid, master_fd = pty.fork()
        if p_pid == 0:  # Child.
            os.execvpe(executable, [executable],
                       env=dict(TERM="linux", COLUMNS="80", LINES="24"))
            os.exit(0)
        else:
            self.screen = pyte.Screen(80, 24)
            self.stream = pyte.Stream(self.screen)
            self.master_fd = master_fd
            print("Zmartwychwstanie!")
            self.state_history.clear()
            self.scores += [0]
            time.sleep(0.1)

        obs = self.get_frame()
        return np.zeros([SCREEN_HEIGHT, SCREEN_WIDTH, len(self.CHAR_BINS)])

    def render(self):
        print("Action input: " + self.action_history[-1])
        self._dump_screen()
        print("Current reward: {}\tPrev reward: {}\tDiff:{}"
              .format(self.scores[-1], self.scores[-2],
                      self.scores[-1] - self.scores[-2]))

    def step(self, action):
        time.sleep(0.001)
        self.tick += 1

        game_input = self.GAME_ACTIONS[action]
        self._push_action(game_input)
        self.action_history += [game_input]

        read_list, _wlist, _xlist = select.select(
            [self.master_fd], [], [], 0)
        while read_list:
            data = os.read(read_list[0], 1024)
            self.stream.feed(data.decode("ascii"))
            read_list, _wlist, _xlist = select.select(
                [self.master_fd], [], [], 0)

        # No more output, let's process results
        if not self.tick % SCREEN_DUMP_RATE:
            self.render()

        self._push_action(" ")
        self.state = self.screen.display
        self.state_history.append(self.state)
        reward = self._calculate_reward()
        self.scores += [reward]

        obs = self.get_frame()
        terminal = self._is_terminal()
        if terminal:
            self.reset()

        info = {}

        return obs, reward, terminal, info


def init_game():
    p_pid, master_fd = pty.fork()
    if p_pid == 0:  # Child.
        os.execvpe(executable, [executable],
                   env=dict(TERM="linux", COLUMNS="80", LINES="24"))
    else:
        wrapper = Wrapper()
        wrapper.run_in_loop()


def init_multiple_games(threads_num):
    for i in range(threads_num):
        p_pid = os.fork()
        if p_pid == 0:  # Child
            init_game()
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Please pass the path to Rogue executable as an argument")
    elif not os.path.isfile(sys.argv[1]):
        sys.exit("Wrong path passed, try again")
    else:
        executable = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2].isdigit:
        threads = int(sys.argv[2])
    else:
        threads = 1

    screen = pyte.Screen(80, 24)
    if threads == 1:
        init_game()
    else:
        init_multiple_games(threads)
