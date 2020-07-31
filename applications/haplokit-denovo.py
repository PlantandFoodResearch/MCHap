#! /usr/bin/env python3

import sys

from haplokit.application.denovo_assembly import program


def main():
    prog = program.cli(sys.argv)
    for line in prog.compute_lines():
        print(line)


if __name__ == "__main__":
    main()
