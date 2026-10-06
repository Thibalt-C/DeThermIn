"""
Command line interface for the DeThermIn project.
"""

from .scripts import selection, predownload, train_ResUNet, set_params
import sys

def cli():
    if len(sys.argv) < 2:
        print("Usage: python -m DeThermIn <command>")
        print("Commands: selection | parameters | train | predownload")
        return

    cmd = sys.argv[1]

    if cmd == "selection":
        selection()
    elif cmd == "parameters":
        set_params()
    elif cmd == "train":
        train_ResUNet()
    elif cmd == "predownload":
        predownload()
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    cli()
