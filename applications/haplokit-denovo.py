#! /usr/bin/env python3

import sys

from haplohelper.application.denovo_assembly import program


def main():
    prog = program.cli(sys.argv)
    vcf = prog.run()
    for line in vcf.lines():
        print(line)


if __name__ == "__main__":
    main()
