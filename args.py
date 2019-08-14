import argparse


def get_train_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Train the model!')

    _add_common_args(parser=parser)

    parser.add_argument("--hidden_size",
                        type=int,
                        required=False,
                        default=100,
                        help="size of hidden layers")
    parser.add_argument("--num_epochs",
                        type=int,
                        required=False,
                        default=50,
                        help="the number of epochs to train for")
    parser.add_argument("--batch_size",
                        type=int,
                        required=False,
                        default=4,
                        help="batch size to train on")
    parser.add_argument("--random_seed",
                        type=int,
                        required=False,
                        default=3716,
                        help="random seed to set for repeatable results")
    parser.add_argument("--exp_name",
                        type=str,
                        required=False,
                        default="sentiment analysis training",
                        help="name of the experiment")
    parser.add_argument("--save_dir",
                        type=str,
                        required=False,
                        default="./logs/train",
                        help="Folder to save the data to")

    args = parser.parse_args()
    return args

def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Download and pre-process netflix review data')

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
    parser.add_argument("--review_limit",
                        type=int,
                        required=False,
                        default=54,
                        help="limit on the number of words to keep from a review")