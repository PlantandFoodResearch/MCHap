import os
import tomllib
from pathlib import Path
from mchap.version import __version__


def test_version():
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path.parent.parent / "pyproject.toml"
    with open(path.resolve(), "rb") as f:
        data = tomllib.load(f)
    toml_version = data["project"]["version"]
    assert __version__ == "v" + toml_version
