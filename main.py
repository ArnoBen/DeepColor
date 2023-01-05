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
    parser.add_argument("-t", "--train", required=False, action="store_true", default=False,
                        help="train a model")
    parser.add_argument("-m", "--model", required=False, default="unet",
                        help="choose the model to train between 'fusion', 'unet', 'gan'")
    return parser


def main():
    parser = parse_arguments()
    args = vars(parser.parse_args())
    log_level = logging.DEBUG if args.pop("debug") else logging.INFO
    logging.basicConfig(filename="logs.txt", level=log_level,
                        format='%(filename)s-%(asctime)s %(levelname)s:%(message)s')


if __name__ == "__main__":
    main()
