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
    parser.add_argument("--word_vecs",
                        type=str,
                        required=False,
                        default="./data/glove.840B.300d.txt",
                        help="File containing the raw negative data")
    parser.add_argument("--review_limit",
                        type=int,
                        required=False,
                        default=54,
                        help="limit on the number of words to keep from a review")


    args = parser.parse_args()
    return args

def _add_common_args(parser):

    parser.add_argument("--clean_train_data",
                        type=str,
                        required=False,
                        default="./data/clean_train.npz",
                        help="the cleaned and featurized train data")
    parser.add_argument("--clean_test_data",
                        type=str,
                        required=False,
                        default="./data/clean_test.npz",
                        help="the cleaned and featurized test data")
    parser.add_argument("--logging_dir",
                        type=str,
                        required=False,
                        default="./logs",
                        help="folder for containing the log data")