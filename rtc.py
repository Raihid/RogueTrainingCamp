#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

class Wrapper():

    actions = {"skip": " ",
               "go_up": "k",
               "go_down": "j",
               "go_left": "h",
               "go_right": "l",
               "eat": "ea"
               # "rest": ".",
               # "search": "s",
               # "descend": ">",
               # "ascend": "<",
               }

    def __init__(self, master_fd, screen):
        self.master_fd = master_fd
        self.screen = screen
        self.stream = pyte.Stream(self.screen)
        self.state = []
        self.rewards = [0]
        self.action_history = ["EPOCH START"]

    def _get_input(self):
        return random.choice(list(self.actions.values()))

    def _dump_screen(self):
        for idx, line in enumerate(self.screen.display, 1):
                print("{0:2d} {1} ¶".format(idx, line))

    def _dump_tick(self):
        print("Action input: " + self.action_history[-1])
        self._dump_screen()
        print("Current reward: {}\tPrev reward: {}\tDiff:{}"
              .format(self.rewards[-1], self.rewards[-2],
                      self.rewards[-1] - self.rewards[-2]))

    def _scrap_screen(self):
        info_line = self.state[-1]

        heuristics = {}
        heuristics["dungeon_level"] = int(re.search("Level: (\d+)", info_line).group(1))
        heuristics["gold"] = int(re.search("Gold: (\d+)", info_line).group(1))

        hp_captures = re.search("Hp: (\d+)\((\d+)\)", info_line)
        heuristics["hp"] = int(hp_captures.group(1))
        heuristics["max_hp"] = int(hp_captures.group(2))

        exp_captures = re.search("Exp: (\d+)/(\d+)", info_line)
        heuristics["exp_level"] = int(exp_captures.group(1))
        heuristics["exp_points"] = int(exp_captures.group(2))

        return heuristics

    def _calculate_reward(self):
        heuristics = self._scrap_screen()
        return (heuristics["gold"] * 5
                + heuristics["hp"] * 100
                + heuristics["dungeon_level"] * 1000)

    def run(self):
        tick = 0
        time.sleep(0.2)
        while True:
            read_list, _wlist, _xlist = select.select(
                [self.master_fd], [], [], 0)

            if not read_list:
                # No more output, let's process results
                tick += 1
                self.state = self.screen.display
                self.rewards += [self._calculate_reward()]
                self._dump_tick()

                input_pyte = self._get_input()
                for input_char in input_pyte:
                    os.write(self.master_fd, bytearray(input_char, "ascii"))
                    time.sleep(0.01)

                self.action_history += [input_pyte]

            else:
                data = os.read(read_list[0], 1024)
                self.stream.feed(data.decode("ascii"))


def init_game():
    screen = pyte.Screen(80, 24)
    p_pid, master_fd = pty.fork()
    if p_pid == 0:  # Child.
        os.execvpe(executable, [executable],
                   env=dict(TERM="linux", COLUMNS="80", LINES="24"))
    else:
        wrapper = Wrapper(master_fd, screen)
        wrapper.run()


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
