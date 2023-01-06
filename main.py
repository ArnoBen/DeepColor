import argparse
import logging


def parse_arguments():
    """
    Parse the cli arguments given by the user.
    Returns:
        parser: parser object containing arguments and their values
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", required=False, action="store_true", default=False,
                        help="set log level to debug")
    parser.add_argument("-t", "--train", required=False, default="unet",
                        help="Train a model. Choose between 'fusion', 'unet', 'gan'.")
    parser.add_argument("-c", "--colorize", required=False,
                        help="Colorize a picture at the given path")
    parser.add_argument("-v", "--video", required=False,
                        help="Colorize a video at the given path")
    return parser


def main():
    parser = parse_arguments()
    parser.print_help()
    args = vars(parser.parse_args())
    log_level = logging.DEBUG if args.pop("debug") else logging.INFO
    logging.basicConfig(filename="logs.txt", level=log_level,
                        format='%(filename)s-%(asctime)s %(levelname)s:%(message)s')


if __name__ == "__main__":
    main()
