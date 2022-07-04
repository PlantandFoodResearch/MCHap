import pytest
import pathlib
import subprocess


@pytest.mark.parametrize(
    "subtool",
    [
        "assemble",
        "call",
        "call-exact",
    ],
)
def test_help_text(subtool):

    path = pathlib.Path(__file__).parent.parent.parent.absolute()
    helpfile = str(path / "cli-{}-help.txt".format(subtool))
    with open(helpfile, "r") as f:
        expect = f.read().split("\n")
    actual = (
        subprocess.check_output("mchap {} -h".format(subtool), shell=True)
        .decode("utf-8")
        .split("\n")
    )
    for exp, act in zip(expect, actual):
        assert exp == act
