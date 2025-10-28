"""Console script for cbcbeatx."""

import argparse


def main():
    """Console script for cbcbeatx."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("_", nargs="*")
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into cbcbeatx.cli.main")
    return 0
