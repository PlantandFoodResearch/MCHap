from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mchap")
except PackageNotFoundError:
    # package is not installed
    pass
