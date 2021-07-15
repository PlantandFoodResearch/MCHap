import sys
import argparse

from mchap.application import assemble
from mchap.application import call
from mchap.application import call_exact
from mchap.application import pedigraph


def main():
    parser = argparse.ArgumentParser(
        "Bayesian assemby of micro-haplotypes in polyploids"
    )

    subprograms = ["assemble", "call", "call-exact", "pedigraph"]
    parser.add_argument(
        "program", nargs=1, choices=subprograms, help="Specify sub-program"
    )
    if len(sys.argv) < 2:
        parser.print_help()

    else:
        args = parser.parse_args(sys.argv[1:2])
        prog = args.program[0]
        if prog == "assemble":
            prog = assemble.program
            prog.cli(sys.argv).run_stdout()
        elif prog == "call":
            prog = call.program
            prog.cli(sys.argv).run_stdout()
        elif prog == "call-exact":
            prog = call_exact.program
            prog.cli(sys.argv).run_stdout()
        elif prog == "pedigraph":
            prog = pedigraph.program
            prog.cli(sys.argv).run()
        else:
            assert False
