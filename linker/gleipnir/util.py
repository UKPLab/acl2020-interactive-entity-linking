import logging
import sys
import logreset


def get_logger(name: str, level=logging.DEBUG):
    logreset.reset_logging()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger