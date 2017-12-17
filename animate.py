#!/usr/bin/env python3
import sys
import time
import os

def is_death_line(line):
    return line == "Zmartwychwstanie!\n"

try: 
    f = open(sys.argv[1]) 
    life_to_record = int(sys.argv[2])
except IndexError:
    print("Please write the name of the file and the number of life you want to animate.")
    sys.exit(1)
except IOError:
    print("Please provide a correct filename.")
    sys.exit(1)
except ValueError:
    print("Please provie a valid life number.")
    sys.exit(1)


life_counter = 0
for line in f:
    if life_counter == life_to_record - 1:
        break
    if is_death_line(line):
        life_counter += 1
else:
    print("There aren't enough lifes, yknow.", life_counter)

life_lines = []

for line in f:
    if is_death_line(line):
        break
    if "¶" in line:  
        life_lines += [line]

while True:
    for idx in range(0, len(life_lines), 24):
        screen = life_lines[idx:idx + 24]
        os.system('clear')
        output = "".join(screen)
        print(output, end='')
        time.sleep(0.25)
