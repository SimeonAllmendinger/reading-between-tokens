import argparse
import torch
import os
import json
import wandb
import warnings
import deepspeed

from fnmatch import fnmatch
from datetime import datetime

from components import (
    data_handler,
    inference,
    probing,
    value_vector_extraction,
    key_value_extraction,
)
from schemas import Issues, Dataset, Probing
from settings import SETTINGS, LOGGER

parser = argparse.ArgumentParser(
    prog="reading-between-tokens",
)

parser.add_argument("--path_dir", type=str, help="Path to the storage directory")
# Deepspeed launcher passes --local_rank; accept it to avoid argparse errors.
parser.add_argument(
    "--local_rank",
    type=int,
    default=int(os.environ.get("LOCAL_RANK", 0)),
    help="Local rank passed by DeepSpeed",
)
parser = deepspeed.add_config_arguments(parser)

class RunHandler():
    """
    Handler for running the reading between tokens pipeline.
    """
    
    def __init__(
        self, 
        path_dir: str,
        country_en: str,
        model_id: str | list[str],
        age: str | list[str],
        gender: str | list[str],
        education: str | list[str],
        hhincome: str | list[str],
        employment: str | list[str],
        left_leaning: str | list[str],
        immigration: str | list[str],
        inequality: str | list[str],
        year_of_election: str | list[int],
        country: str | list[str],
        ) -> None:
        
        # Directories
        self.path_data_dir = os.path.join(path_dir, "data")
        self.path_activation_dir = os.path.join(path_dir, "activations")
        self.path_probing_dir = os.path.join(path_dir, "probing")
        self.path_value_vector_dir=os.path.join(path_dir, "value_vectors")
        self.path_personas_dir=os.path.join(path_dir, "personas")
        self.path_key_values_dir=os.path.join(path_dir, "key_values")
        
        # Sweep parameters
        self.country = country_en if isinstance(country_en, str) else country_en[0]
        self.model_id = model_id if isinstance(model_id, str) else model_id[0]
        self.persona_config = {
            "age": age,
            "gender": gender,
            "education": education,
            "hhincome": hhincome,
            "employment": employment,
            "left_leaning": left_leaning,
            "immigration": immigration,
            "inequality": inequality,
            "year_of_election": year_of_election,
            "country": country,
        }
        
        LOGGER.info(f"Country: {self.country}")
        LOGGER.info(f"Model: {self.model_id}")
        LOGGER.info(f"Persona Config: {self.persona_config}")

    def get_dataset(self) -> Dataset:
        """
        Get the dataset.
        
        Returns:
        - dataset: Dataset
            The dataset.
        """
        
        # Check if the dataset should be loaded from an existing dataset
        if SETTINGS.dataset['FROM_EXISTING_DATASET']:
            
            # Check if the dataset is already available
            if SETTINGS.dataset['PATH_DATASET'] is None:
                
                # Get the newest dataset file
                path_dataset_dir = os.path.join(
                    self.path_data_dir,
                    self.country.lower(),self.model_id
                )
                dataset_files = [f for f in os.listdir(path_dataset_dir) if fnmatch(f, '*_dataset_*.json')]
                
                try:
                    # Sort files by extracting the datetime from the filename
                    dataset_files.sort(key=lambda x: datetime.strptime(x[:19], '%Y-%m-%d_%H-%M-%S'), reverse=True)
                    newest_dataset_file = dataset_files[0]
                except IndexError:
                    raise FileNotFoundError("No dataset files found.")
                
                path_dataset = os.path.join(path_dataset_dir, newest_dataset_file)
            else: 
                # use the path from the settings
                path_dataset = SETTINGS.dataset['PATH_DATASET']
                
            with open(path_dataset, "r") as file:
                dataset_items = json.load(file)
            
            _dataset = Dataset(**dataset_items)
                
        else:
            # Create the issues
            LOGGER.info("Creating issues ...")
            
            issue_items, issue_config = data_handler.get_issue_items(
                path_data_dir=self.path_data_dir,
                country=self.country
            )

            issues = Issues(
                country=self.country,
                data_dir=self.path_data_dir,
                issue_items=issue_items,
                issue_config=issue_config
            )
            # save the issues
            issues.save_as_json()

            data = inference.get_inference_items(
                issues=issues,
                path_activation_dir=self.path_activation_dir,
                model_id=self.model_id
            )
            _dataset = Dataset(
                data=data,
                data_dir=self.path_data_dir,
                inference_config=SETTINGS.inference,
                issue_config=issue_config,
                country=self.country
            )
        
            # save the dataset
            _dataset.save_as_json()
        
        LOGGER.info(f"Issue Owner names in Dataset: {_dataset.issue_owner_names}")
        
        return _dataset

    def get_probing(self, dataset: Dataset) -> Probing:
        """
        Get the probing.
        
        Args:
        - dataset: Dataset
            The dataset.
        
        Returns:
        - probing: Probing
            The probing.
        """
        
        if SETTINGS.probing['FROM_EXISTING_PROBING']:
            
            path_probing_dir = os.path.join(self.path_probing_dir,
                                            self.country.lower(),
                                            self.model_id)
            
            probing_files = [f for f in os.listdir(path_probing_dir) if fnmatch(f, '*_probing_*.json')]
            
            try:
                # Sort files by extracting the datetime from the filename
                probing_files.sort(key=lambda x: datetime.strptime(x[:19], '%Y-%m-%d_%H-%M-%S'), reverse=True)
                newest_probing_file = probing_files[0]
            except IndexError:
                raise FileNotFoundError("No probing files found.")
            
            LOGGER.info(f"Loading probing from {newest_probing_file} ...")
            
            path_probing = os.path.join(
                path_probing_dir,
                newest_probing_file
            )
            
            with open(path_probing, "r") as file:
                probing_json = json.load(file)
                
            _probing = Probing(**probing_json)
        
        else:
            # Get the probing results
            value_probes = probing.get_value_probes(
                dataset=dataset,
                path_value_probe_dir=self.path_probing_dir
            )
            
            _probing = Probing(
                value_probes=value_probes,
                path_probing_dir=self.path_probing_dir,
                probing_config=SETTINGS.probing,
                country=self.country,
            )
            
            _probing.save_as_json()
            
        return _probing
    
    def get_value_vector_extraction(self, probing: Probing, dataset: Dataset) -> Probing:
        """
        Get the value vectors.
        
        Args:
        - probing: Probing
            The probing.
        
        Returns:
        - probing: Probing
            The probing.
        """
        
        if probing.value_vectors is not None:
            return probing
        
        # Extract the value vectors
        value_vector_extractor = value_vector_extraction.ValueVectorExtractor(
            probing=probing,
            dataset=dataset,
            path_value_vector_dir=self.path_value_vector_dir
        )
        _value_vectors = value_vector_extractor.extract()
        
        probing.value_vectors = _value_vectors
        
        probing.save_as_json()
        
        return probing

    def key_value_extraction(self, probing: Probing):
        
        LOGGER.info(f"io_name_mapped: {SETTINGS.dataset[self.country]['name_mapping'].keys()}")
        
        key_value_extractor = key_value_extraction.KeyValueExtractor(
            path_key_values_dir=self.path_key_values_dir,
            path_personas_dir=self.path_personas_dir,
            persona_config=self.persona_config,
            model_id=self.model_id,
            country=self.country,
            probing=probing,
        )
        key_value_extractor.extract()


def run(path_dir: str, run_config: dict) -> None:
    """
    Run the reading between tokens pipeline.

    Args:
    - path_dir: str
        The path to the storage directory.
    - run_config: dict
        The run configuration.
    """

    # Start application
    LOGGER.info("Starting ...")
    LOGGER.info(f"Available GPU devices: {torch.cuda.device_count()}")
    LOGGER.info(f"Run Config: {run_config}")
    
    persona_config = SETTINGS.personas[run_config['country_en'][0]]['PARAMETERS']
    run_config.update(persona_config)
    
    run_handler = RunHandler(
        path_dir=path_dir,
        **run_config)
    
    ###########
    # DATASET #
    ###########
    
    _dataset = run_handler.get_dataset()
        
    ###########
    # PROBING # 
    ###########
        
    _probing = run_handler.get_probing(dataset=_dataset)
        
    ###########################
    # VALUE VECTOR EXTRACTION #
    ###########################
    
    _probing_with_value_vectors = run_handler.get_value_vector_extraction(probing=_probing, dataset=_dataset)
    
    ########################
    # KEY VALUE EXTRACTION #
    ########################
           
    run_handler.key_value_extraction(probing=_probing_with_value_vectors)
    

def main(run_config: dict = None):
    """
    Main function to run the reading between tokens pipeline.
    
    Args:
    - run_config: dict
        The run configuration.
    """
    args = parser.parse_args()
    path_dir = args.path_dir
    
    run(path_dir, run_config)
