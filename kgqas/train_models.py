#!/usr/bin/env python3

# Trains the models

# Notes:
# Training takes about 10 minutes on a moderately spec'd laptop.
# Uses about 700 MiB of memory during training. 


import time

from loguru import logger

from semantic_parsing import SemanticParsingClass


def time_in_words(sec: float) -> str:
    """
    Convert seconds into a word reading for time.  It does not do proper pluralization.
    """
    if sec < 60:
        return f'{sec} seconds'
    elif sec < 3600:
        minutes = sec // 60
        seconds = sec % 60
        return f'{int(minutes)} minutes {int(seconds)} seconds'
    else:
        hours = sec // 3600
        minutes = sec % 3600 // 60
        seconds = sec % 60
        return f'{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds'


if __name__ == '__main__':
    elapsed = -time.time()
    sem_par = SemanticParsingClass()
    sem_par.train_all()
    elapsed += time.time()

    logger.info(f'Total time elapsed during training: {time_in_words(elapsed)}')

