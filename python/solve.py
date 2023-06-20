import argparse
from loader import Simulation, compile
from solvers import find_end
from shutil import which


def main():
    parser = argparse.ArgumentParser(
        "run", description="runs a action of the simulation"
    )
    parser.add_argument(
        "--wrapper",
        "-w",
        type=str,
        nargs="?",
        help="path to wrapper to load",
        default="wrapper.py",
    )
    parser.add_argument(
        "--rlc",
        "-c",
        type=str,
        nargs="?",
        help="path to rlc compiler",
        default="rlc",
    )
    parser.add_argument(
        "--source",
        "-rl",
        type=str,
        nargs="?",
        help="path to .rl source file",
    )
    parser.add_argument("--show-actions", "-a", action='store_true', default=False)

    args = parser.parse_args()
    assert (
        which(args.rlc) is not None
    ), "could not find executable {}, use --rlc <path_to_rlc> to configure it".format(
        args.rlc
    )

    sim = (
        Simulation(args.wrapper[0])
        if len(args.wrapper) == 1
        else compile(args.source, args.rlc)
    )

    if args.show_actions is not None:
        sim.dump()
        return

    state = sim.start(["play"])
    state.dump()
    find_end(sim, state)


if __name__ == "__main__":
    main()
