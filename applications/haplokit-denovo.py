#! /usr/bin/env python3

import sys

from haplokit.application.denovo_assembly import program


def main():
    prog = program.cli(sys.argv)
    prog.run_stdout()


if __name__ == "__main__":
    main()
