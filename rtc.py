#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import pty
import pyte
import random
import re
import select
import sys
import time

# TODO
# Pobawić się jakimś agentem
# Pomyśleć o heurystykach do eksploracji
# Uproszczenie akcji
# Gif/animacja
# Sygnał po zakończeniu tury od rogue
# Obraz nie jest kwadratowy D:


executable = "/home/rahid/Programowanie/Repos/rogue5.2/rogue"


class Wrapper():
    GAME_ACTIONS = [# " ",  # skip
                    "k",  # go_up
                    "j",  # go_down
                    "h",  # go_left
                    "l",  # go_right
                    ">",  # descend
                    "s",  # search
                    # "." search
                    ]
    ACTIONS = len(GAME_ACTIONS)
    NAME = "Rogue"

    def __init__(self):
        self.state = []
        self.rewards = [0]
        self.action_history = ["EPOCH START"]
        self.tick = 0
        self.last_death = 0

        self._start_game()

    def _get_input(self):
        return random.choice(self.GAME_ACTIONS)

    def _dump_screen(self):
        for idx, line in enumerate(self.screen.display, 1):
                print("{0:2d} {1} ¶".format(idx, line))

    def _dump_tick(self):
        print("Action input: " + self.action_history[-1])
        self._dump_screen()
        print("Current reward: {}\tPrev reward: {}\tDiff:{}"
              .format(self.rewards[-1], self.rewards[-2],
                      self.rewards[-1] - self.rewards[-2]))

    def score_for_logs(self):
        score = self.rewards[-1] - self.rewards[self.last_death]
        self.last_death = self.tick
        return score

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
        heuristics["dungeon_level"] = int(re
                                          .search("Level:\s+(\d+)", info_line)
                                          .group(1))
        heuristics["gold"] = int(re.search("Gold:\s+(\d+)", info_line).group(1))

        hp_captures = re.search("Hp:\s+(\d+)\((\d+)\)", info_line)
        heuristics["hp"] = int(hp_captures.group(1))
        heuristics["max_hp"] = int(hp_captures.group(2))

        exp_captures = re.search("Exp:\s+(\d+)/(\d+)", info_line)
        heuristics["exp_level"] = int(exp_captures.group(1))
        heuristics["exp_points"] = int(exp_captures.group(2))

        heuristics["explored"] = sum(not c.isspace()
                                     for line in self.state[1:-1] for c in line)

        return heuristics

    def _calculate_reward(self):
        heuristics = self._scrap_screen()
        return (heuristics["explored"]
                + (heuristics["dungeon_level"] - 1) * 1000)

    def _push_action(self, input_):
        for input_char in input_:
            os.write(self.master_fd, bytearray(input_char, "ascii"))
            time.sleep(0.01)

    def get_frame(self):
        state_as_int = []
        padding = (80 - 24) // 2
        for _ in range(padding):
            state_as_int += [0 for _ in range(80)]

        for line in self.state:
            state_as_int += [ord(c) for c in line]

        for _ in range(padding):
            state_as_int += [0 for _ in range(80)]
        return np.array(state_as_int)

    def run_in_loop(self):
        tick = 0
        time.sleep(0.2)

        while True:
            read_list, _wlist, _xlist = select.select(
                [self.master_fd], [], [], 0)

            if not read_list:
                # No more output, let's process results
                terminal = self._is_terminal()
                if terminal:
                    self._start_game(restart=True)
                    continue

                tick += 1
                self.state = self.screen.display
                self.rewards += [self._calculate_reward()]
                self._dump_tick()
                game_input = self._get_input()
                self._push_action(game_input)
                self.action_history += [game_input]
            else:
                data = os.read(read_list[0], 1024)
                self.stream.feed(data.decode("ascii"))

    def _is_terminal(self):
        screen = "".join(line for line in self.screen.display)
        return ("REST" in screen and "IN" in screen and "PEACE" in screen)

    def _start_game(self, restart=False):
        if restart:
            self._push_action(" ")
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
            time.sleep(0.2)

    def next_tick(self, action):
        time.sleep(0.01)
        self.tick += 1

        game_input = self.GAME_ACTIONS[action]
        self._push_action(game_input)
        self.action_history += [game_input]

        while True:
            read_list, _wlist, _xlist = select.select(
                [self.master_fd], [], [], 0)

            if not read_list:
                if not self.tick % 10:
                    self._dump_tick()

                terminal = self._is_terminal()
                self._push_action(" ")
                if terminal:
                    self.rewards += [self.rewards[-1]]
                    self._start_game(restart=True)
                    break
                # No more output, let's process results
                self.state = self.screen.display
                self.rewards += [self._calculate_reward()]
                break

            else:
                data = os.read(read_list[0], 1024)
                self.stream.feed(data.decode("ascii"))

        return self.rewards[-1] - self.rewards[-2], terminal


def init_game():
    screen = pyte.Screen(80, 24)
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
