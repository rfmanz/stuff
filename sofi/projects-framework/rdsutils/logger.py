import sys, os
import logging
import datetime as dt


def get_file_stdout_logger(log_dir, log_file, append_dir_timestamp=True):
    """
    Get default logger that both saves to file and log to stdout
    
    @params log_dir: directory to store the log
    @params log_file: name of the log
    @params append_dir_timestamp: append timestamp if True as ID
    
    @returns logger: the logger object
    """
    # set dir
    if append_dir_timestamp:
        tstamp = int(dt.datetime.now().timestamp())
        log_dir = f"{log_dir}_{tstamp}"
    os.makedirs(log_dir, exist_ok=True)    
    log_path = os.path.join(log_dir, log_file)
    
    # build logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', 
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
        
    # to avoid outputing multiple times to stdout
    if not logger.hasHandlers():
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)    
    logger.addHandler(file_handler) 
    
    return logger


def get_file_logger(log_dir, log_file, append_dir_timestamp=True):
    """
    Get default logger that both saves to file and log to stdout
    
    @params log_dir: directory to store the log
    @params log_file: name of the log
    @params append_dir_timestamp: append timestamp if True as ID
    
    @returns logger: the logger object
    """
    
    # set dir
    if append_dir_timestamp:
        tstamp = int(dt.datetime.now().timestamp())
        log_dir = f"{log_dir}_{tstamp}"
    os.makedirs(log_dir, exist_ok=True)    
    log_path = os.path.join(log_dir, log_file)
    
    # build logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', 
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger