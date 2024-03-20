# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional
import logging


logger_initialized = {}

def get_logger(
        name: str = 'DiffBindFR',
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        io_mode: str = 'w'
) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, io_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger_initialized[name] = True

    return logger