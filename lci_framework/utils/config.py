"""
Configuration and Logging Utilities

This module provides utilities for managing configuration and setting up logging.
"""

import os
import yaml
import json
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(
    default_level: int = logging.INFO,
    log_config_path: Optional[str] = None,
    log_file: Optional[str] = None
):
    """
    Setup logging configuration
    
    Args:
        default_level: Default logging level
        log_config_path: Path to logging configuration file
        log_file: Path to log output file
    """
    if log_config_path and os.path.exists(log_config_path):
        # Load logging configuration from file
        with open(log_config_path, 'rt') as f:
            if log_config_path.endswith('.yaml') or log_config_path.endswith('.yml'):
                config = yaml.safe_load(f.read())
            else:
                config = json.load(f)
                
        # If log file specified, update the file handlers
        if log_file:
            for handler in config['handlers'].values():
                if 'filename' in handler:
                    handler['filename'] = log_file
                    
        # Apply configuration
        logging.config.dictConfig(config)
    else:
        # Basic configuration
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file
        )
        
    logging.info("Logging configured")
    

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the file format is unsupported
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
            
    logging.info(f"Loaded configuration from {config_path}")
    return config
    

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif config_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
            
    logging.info(f"Saved configuration to {config_path}")
    

def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration
    
    Returns:
        Dictionary with default configuration values
    """
    config = {
        "environment": {
            "n_states": 20,
            "sparse_transitions": True,
            "seed": 42
        },
        "evolution": {
            "pop_size": 50,
            "mutation_rate": 0.1,
            "tournament_size": 5,
            "elitism": 2,
            "n_generations": 50,
            "steps_per_eval": 100,
            "seed": 42
        },
        "agent": {
            "learning_rate_range": [0.001, 0.1],
            "hidden_size_range": [8, 128],
            "n_layers_range": [1, 3],
            "l1_reg_range": [0.0, 0.01],
            "l2_reg_range": [0.0, 0.01],
            "dropout_rate_range": [0.0, 0.5],
            "initial_energy": 1000.0
        },
        "output": {
            "output_dir": "results",
            "save_interval": 10
        }
    }
    
    return config


def default_logging_config() -> Dict[str, Any]:
    """
    Create default logging configuration
    
    Returns:
        Dictionary with default logging configuration
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": "lci_framework.log",
                "mode": "w"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": True
            }
        }
    }
    
    return config 