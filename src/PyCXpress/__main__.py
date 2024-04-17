# pylint: disable=missing-function-docstring

import argparse
import sys
import sysconfig

from pybind11 import get_include as pybind11_include

from PyCXpress import get_include, version


def print_includes() -> None:
    dirs = {
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
        pybind11_include(),
        get_include(),
    }

    print(" ".join(f"-I {d}" for d in dirs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=version,
        help="Print the version and exit.",
    )
    parser.add_argument(
        "--includes",
        action="store_true",
        help="Include flags for both pybind11 and Python headers.",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.includes:
        print_includes()


if __name__ == "__main__":
    main()
