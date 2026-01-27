import os
import logging

from typing import final

from utils import get_project_root, get_logger, get_config

class Settings():
    """
    Class to store settings and configurations
    """

    __path_project_root=get_project_root()
    __logger = get_logger(path_project_root=__path_project_root)
    
    def __init__(self):
        
        # Utils
        self.__logger.debug(f'PATH PROJECT ROOT: {get_project_root()}')
        
        # Config
        self.api_keys = get_config(os.path.join(self.__path_project_root,'configs/utils/config_api_keys.yaml'))
        
        # Dataset
        self.dataset = get_config(os.path.join(self.__path_project_root,'configs/config_dataset.yaml'))
        
        # Inference
        self.inference = get_config(os.path.join(self.__path_project_root,'configs/config_inference.yaml'))
        
        # Probing
        self.probing = get_config(os.path.join(self.__path_project_root,'configs/config_probing.yaml'))
        
        # Value Vector Extraction
        self.value_vector_extraction = get_config(os.path.join(self.__path_project_root,'configs/config_value_vector_extraction.yaml'))
        
        # Key Vector Extraction
        self.key_value_extraction = get_config(os.path.join(self.__path_project_root,'configs/config_key_value_extraction.yaml'))
        
        # Personas
        self.personas = get_config(os.path.join(self.__path_project_root,'configs/config_personas.yaml'))
        
        # Sweep
        self.sweep = get_config(os.path.join(self.__path_project_root,'configs/config_sweep.yaml'))
        
    @staticmethod
    def logger() -> logging.Logger:
        """Function to get logger

        Returns:
            logging.Logger: logger object
        """    
        return Settings.__logger
    
SETTINGS: final = Settings()
LOGGER: final = SETTINGS.logger()
