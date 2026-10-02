"""python -m bouldering <command> [options]"""
import argparse

from .cli import climb, export_onnx, pose, routes, segment, track

COMMANDS = {
    "track": track,
    "routes": routes,
    "climb": climb,
    "segment": segment,
    "pose": pose,
    "export-onnx": export_onnx,
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bouldering", description=__import__("bouldering").__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, module in COMMANDS.items():
        p = sub.add_parser(name, help=module.__doc__.strip().split("\n")[0], description=module.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        module.add_arguments(p)
        p.set_defaults(run=module.run)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
