import os
from pathlib import Path
from logging import Logger, Formatter, FileHandler, basicConfig, INFO, StreamHandler
from typing import Tuple, Dict, List, Union, Optional


LOG_FORMAT_DEFAULT = "[%(asctime)s: %(levelname)s-%(name)s]: %(message)s"


class EmptyLogger(Logger):
    """
    an empty logger with logger API but do not log
    >>> l = EmptyLogger()
    >>> l.info('empty')
    """

    def __init__(self):
        pass

    def info(self, msgm, *arg, **kwargs):
        pass

    def debug(self, msgm, *arg, **kwargs):
        pass

    def exception(self, msgm, *arg, exe_info=True, **kwargs):
        pass

    def error(self, msgm, *arg, **kwargs):
        pass

    def warn(self, msgm, *arg, **kwargs):
        pass


def make_logger(
    log_name: Optional[str],
    log_file: str = None,
    log_level: int = INFO,
    log_format_str: str = LOG_FORMAT_DEFAULT,
    overwrite_file=False,
    **kwargs
):

    if log_name is None:
        return EmptyLogger(), {}

    # initial logger
    log_er = Logger(name=log_name)
    basicConfig()

    # formatter
    formatter = Formatter(fmt=log_format_str)

    # setup file handler
    if log_file is None:
        file_handler = StreamHandler()
    else:
        if overwrite_file and os.path.exists(log_file):
            os.remove(log_file)

        p = Path(log_file)
        p.parents[0].mkdir(parents=True, exist_ok=True)
        file_handler = FileHandler(p)

    file_handler.setFormatter(formatter)
    log_er.addHandler(file_handler)

    # set logging level
    log_er.setLevel(log_level)

    return log_er, kwargs


# logger, kwargs = make_logger(log_name="test",
#                              log_file="../log/test.log",
#                              overwrite_file=True)
