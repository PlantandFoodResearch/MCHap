import sys
import argparse

from mchap.application import assemble
from mchap.application import call
from mchap.application import call_exact
from mchap.application import find_snvs
from mchap.application import call_pedigree
from mchap.application import atomize

from mchap import __version__


def main():
    parser = argparse.ArgumentParser(
        "Bayesian assembly of micro-haplotypes in polyploids"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"mchap {__version__}",
    )

    subprograms = ["assemble", "call", "call-exact", "call-pedigree", "find-snvs", "atomize"]

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
        elif prog == "find-snvs":
            find_snvs.main(sys.argv)
        elif prog == "call-pedigree":
            prog = call_pedigree.program
            prog.cli(sys.argv).run_stdout()
        elif prog == "atomize":
            atomize.main(sys.argv)
        else:
            assert False
