import sys
import argparse

from haplokit.application import denovo
from haplokit.application import pedigraph

def main():
    parser = argparse.ArgumentParser(
        'Bayesian assemby of micro-haplotypes in polyploids'
    )

    subprograms = ['denovo', 'pedigraph']
    parser.add_argument('program',
                        nargs=1,
                        choices=subprograms,
                        help='Specify sub-program')
    if len(sys.argv) < 2:
        parser.print_help()
    
    else:
        args = parser.parse_args(sys.argv[1:2])
        prog = args.program[0]
        if prog == 'denovo':
            prog = denovo.program
        elif prog == 'pedigraph':
            prog = pedigraph.program
        prog.cli(sys.argv)
