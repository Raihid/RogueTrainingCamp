#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pty
import select
import sys

import pyte
import time
import random


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

    def _get_input(self):
        time.sleep(0.5)
        return random.choice(list(self.actions.values()))

    def _screen_dump(self):
        for idx, line in enumerate(self.screen.display, 1):
                print("{0:2d} {1} ¶".format(idx, line))

    def run(self):
        tick = 0
        while 1:
            read_list, _wlist, _xlist = select.select(
                [self.master_fd], [], [], 0)

            if not read_list:
                tick += 1
                self._screen_dump()

                input_pyte = self._get_input()
                # print(pyte_input)
                for input_char in input_pyte:
                    os.write(self.master_fd, bytearray(input_char, "ascii"))
                    time.sleep(0.01)

            else:
                self._screen_dump()
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
