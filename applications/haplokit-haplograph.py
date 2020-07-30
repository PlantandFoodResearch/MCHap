#! /usr/bin/env python3

import sys

from haplokit.application.pedigree_plot import program


def main():
    prog = program.cli(sys.argv)
    prog.run()

if __name__ == "__main__":
    main()
