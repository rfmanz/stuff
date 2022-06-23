from glob import glob
from itertools import chain
from logging import Logger, Formatter, FileHandler, basicConfig, INFO, StreamHandler
from os import remove
from os.path import abspath, basename, dirname, join, exists
import sys
import traceback
from pathlib import Path
from typing import TypeVar, Tuple, Dict, List, Union, Iterable, Optional

T=TypeVar('T')

LOG_FORMAT_DEFAULT='%(asctime)s-%(levelname)s-%(name)s-%(message)s'

class EmptyLogger(Logger): 
    """ 
    an empty loggrt with logger API but do no logging
    >>> l =EmptyLogger()
    >>> l.info('empty')
    """
    def __init__(self):
        """ dummy init method, do nothing """
        pass   
    
    def info(self, msgm, *arg, **kwargs):
        pass
    
    def debug(self, msgm, *arg, **kwargs):
        pass  
    
    def exception(self, msgm, *arg,exc_info=True, **kwargs):
        pass  
    
    def error(self, msgm, *arg, **kwargs):
        pass 
    
    def warn(self, msgm, *arg, **kwargs):
        pass  

    
def make_logger(
    log_name: Optional[str],
    log_file: str= None, 
    log_level: int=INFO, 
    log_format_str: str=LOG_FORMAT_DEFAULT,
    **kwargs
) -> Tuple[Logger, Dict]:
    
    if log_name is None:
        return EmptyLogger(), {}
    
    #initial logger
    log_er=Logger(name=log_name)
    basicConfig()
    
    #setup formatter
  
    formatter=Formatter(fmt=log_format_str)
    
    #setup file handler
    if log_file is None:
        file_handler=StreamHandler()
    else:
        file_handler=FileHandler(log_file)
        
    file_handler.setFormatter(formatter) 
    log_er.addHandler(file_handler)
    
    #setup logging level
    log_er.setLevel(log_level)
    
    return log_er, kwargs

def parse_glob_paths(path: Union[str, List[str], Path, List[Path]], header='') -> List[str]:
    path=None
    if isinstance(path, (str, Path)):
        path_=[str(path),]
    elif isinstance(path, list):
        path_=[str(p) for p in path]
    
    path_=chain(*[p.split(';') for p in path_])
    path_: Iterable[str]=set(path_)
    
    if header:
        path_=[p if p.startwith(header) else header + p for p in path_]
    
    return list(path_)
        
class MultiServerSafeMixin:
    _occupy_file_name: str = '{file_name}__occ_'
    
    def occupy(self, 
               target_file_path: str, 
    ) -> Tuple[bool, str, Optional[str]]:
        if exists(occupy_file):
            return False, occupy_file, "found the occupation file exists"
        if not exists(target_dir_path):
            Path(target_dir_path).mkdir(parents=True, exist_ok=False)
        try:
            Path(occupy_file).touch(exist_ok=False)
        except BaseException as err:
            return False, occupy_file, 'when trying to occupy the process, failed with error %s, %s' % (
            type(err), err)
        return True, occupy_file, None
    
    
def report_error(
    occupy_file: str
):
    remove(occupy_file)
    et, ev, tb=sys.exc_info()
    tb_str=traceback.format_exception(et, ev, tb)
    
    with open(occupy_file+'_ERROR', 'w') as ef:
        ef.writelines(tb_str)
