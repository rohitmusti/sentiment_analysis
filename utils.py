import logging
import os
from tqdm import tqdm

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

    


