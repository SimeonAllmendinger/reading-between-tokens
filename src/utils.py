import os
import logging
import yaml
from datetime import datetime

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_logger(path_project_root: str) -> logging.Logger:
    log_dir = os.path.join(path_project_root, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(
        log_dir,
        f"master-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    config_path = os.path.join(path_project_root, 'configs/utils/config_logger.conf')
    logging.config.fileConfig(
        config_path,
        defaults={'logfilename': log_filename},
        disable_existing_loggers=False
    )
    master_logger = logging.getLogger('master')
    return master_logger

def get_config(path_config_file: str) -> dict:
    with open(path_config_file,'r') as file:
        config = yaml.safe_load(file)
    return config
