import deepspeed

from settings import SETTINGS
from main import main

def run() -> None:
    """
    Run the main function with the given arguments.
    """
    
    parameters = SETTINGS.sweep.get("parameters", {})
    run_config = {key: param.get("values", []) for key, param in parameters.items()}
    main(run_config=run_config)
    
if __name__ == "__main__":
    run()
    
    
    