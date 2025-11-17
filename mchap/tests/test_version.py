import os
from pathlib import Path
from mchap.version import __version__


def test_version():
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path.parent.parent / "pyproject.toml"
    with open(path.resolve(), "rb") as f:
        for line in f.readlines():
            line = line.decode()
            if line.startswith("version"):
                toml_version = line.split('"')[1]
                break
    assert __version__ == "v" + toml_version
