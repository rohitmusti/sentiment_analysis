import argparse


def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Download and pre-process SQuAD')

    _add_common_args(parser=parser)

    parser.add_argument("--raw_pos",
                        type=str,
                        required=False,
                        default="./data/rt-polarity.pos",
                        help="File containing the raw positive data")
    parser.add_argument("--raw_neg",
                        type=str,
                        required=False,
                        default="./data/rt-polarity.neg",
                        help="File containing the raw negative data")

    args = parser.parse_args()
    return args

def _add_common_args(parser):
    pass

