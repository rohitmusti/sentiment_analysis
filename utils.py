import torch
import torch.utils.data as data
import numpy as np
import torch
import logging
import os
from tqdm import tqdm

class netflix_reviews(data.Dataset):
    def __init__(self, data_path):
        dataset = np.load(data_path)
        self.rw_tokens = torch.from_numpy(dataset['rwtokens']).long()
        self.sentiment = torch.from_numpy(dataset['sentiment'])
        self.ids = torch.from_numpy(dataset['ids'])

    def __getitem__(self, idx):

        example = (
            self.ids[idx],
            self.rw_tokens[idx],
            self.sentiment[idx]
        )

        return example
    
    def __len__(self):
        return len(self.ids)

def collate_fn(examples):

    def merge_0d(scalars, dtype=torch.long):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.long, pad_value=0, pad_length=54):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), pad_length, 300, dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end,] = seq[:end]
        return padded, torch.tensor(lengths)
    
    ids, rw_tokens, sentiments = zip(*examples)

    ids = merge_0d(ids)
    rw_tokens, lengths = merge_1d(rw_tokens)
    sentiments = merge_0d(sentiments, dtype=torch.bool)

    return (ids, rw_tokens, sentiments, lengths)

def data_importer(datapath):
    with open(file=datapath, mode="r", encoding="Windows-1252") as fh:
        source = fh.readlines()
    return source

def get_logger(logging_dir, name):
    """
    please see: https://github.com/chrischute/squad/blob/master/util.py
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(logging_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

    


