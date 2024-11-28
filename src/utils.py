import datetime
import os
import yaml

def save_versioned_model(model, target_name, metrics, logger=None):
    """
    Saves the model with versioning and updates the model registry.

    Parameters:
    - model: The trained model to save.
    - target_name (str): The name of the target model (eg. "response_type").
    - metrics (dict): Dictionary containing evaluation metrics.
    - logger (Logger, optional): Logger instance for logging actions.
    """
    # Generate versioned filename by using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    versioned_model_path = f"../models/lstm_models/{target_name}_model_{timestamp}.h5"

    # Save the model
    model.save(versioned_model_path)
    if logger:
        logger.info(f"Model for {target_name} saved to {versioned_model_path}")
    else:
        print(f"Model for {target_name} saved to {versioned_model_path}")

    # Update the model registry
    update_model_registry(target_name=target_name, model_path=versioned_model_path, metrics=metrics, logger=logger)

def update_model_registry(target_name, model_path, metrics, logger=None):
    """
    Updates the model registry with the latest model information.

    Parameters:
    - target_name (str): The name of the target model (eg. "response_type").
    - model_path (str): The file path of the saved model.
    - metrics (dict): Dictionary containing evaluation metrics.
    - logger (Logger, optional): Optional logger for logging info or warnings.
    """
    # Load the existing registry or create a new one
    model_registry_path = "../models/model_registry.yaml"
    if os.path.exists(model_registry_path):
        with open(model_registry_path, "r") as f:
            registry = yaml.safe_load(f) or {}
    else:
        registry = {}

    # Add the entry for the target model
    if target_name not in registry:
        registry[target_name] = []

    # Create the entry for the new model version
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics = {key: float(value) for key, value in metrics.items()}
    new_entry = {
        "version": timestamp,
        "path": model_path,
        "metrics": metrics,
    }
    registry[target_name].append(new_entry)

    # Save the updated registry
    with open(model_registry_path, "w") as f:
        yaml.dump(registry, f)

    if logger:
        logger.info(f"Updated model registry for {target_name} with version {timestamp}")
    else:
        print(f"Updated model registry for {target_name} with version {timestamp}")

def load_latest_model_path(target_name, logger=None):
    """
    Loads the path of the latest version of the model for target_name from the model registry.

    Parameters:
    - target_name (str): The name of the target model (eg. "response_type").
    - logger (Logger, optional): Logger instance for logging actions.

    Returns:
    - str: Path to the latest model version.
    """
    model_registry_path = "../models/model_registry.yaml"
    if not os.path.exists(model_registry_path):
        raise FileNotFoundError(f"Model registry file not found at {model_registry_path}")

    with open(model_registry_path, "r") as f:
        registry = yaml.safe_load(f)

    # Get the latest version of the model for the target
    if target_name in registry:
        latest_model = sorted(registry[target_name], key=lambda x: x["version"], reverse=True)[0]
        if logger:
            logger.info(f"Loading model from {latest_model['path']}")
        else:
            print(f"Loading model from {latest_model['path']}")
        return latest_model["path"]
    else:
        raise ValueError(f"No models found for target: {target_name}")
