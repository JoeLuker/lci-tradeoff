#!/usr/bin/env python3
"""
Main entry point for running the LCI framework with MLX-based implementation.
This implementation utilizes Apple Silicon acceleration for faster simulations.
"""

import os
import argparse
import yaml
import logging
from datetime import datetime
from lci_framework.core.evolution import TensorEvolution

def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_simulation(config):
    """Run a tensor-based LCI simulation with the given configuration."""
    # MLX automatically uses the most efficient device available on Apple Silicon
    print("Using MLX with unified memory architecture for acceleration")
    
    # Create output directory
    output_dir = config.get('output_dir', 'results/lci_run')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the evolution
    evolution = TensorEvolution(
        pop_size=config.get('pop_size', 100),
        n_states=config.get('n_states', 16),
        n_actions=config.get('n_actions', 4),
        hidden_size=config.get('hidden_size', 64),
        n_layers=config.get('n_layers', 2),
        dropout_rate=config.get('dropout_rate', 0.1),
        mutation_rate=config.get('mutation_rate', 0.1),
        tournament_size=config.get('tournament_size', 5),
        elitism=config.get('elitism', 0.1),
        energy_init=config.get('energy_init', 10.0),
        energy_cost_predict=config.get('energy_cost_predict', 0.01),
        energy_cost_learn=config.get('energy_cost_learn', 0.05),
        energy_recovery=config.get('energy_recovery', 0.1),
        output_dir=output_dir
    )
    
    # Run the simulation
    evolution.run_simulation(
        n_generations=config.get('n_generations', 50),
        steps_per_generation=config.get('steps_per_generation', 200),
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    # Save results
    evolution.save_results()
    evolution.plot_results()
    
    print(f"Simulation completed. Results saved to {output_dir}")
    return output_dir

def main(args=None):
    """Main entry point."""
    if args is None:
        parser = argparse.ArgumentParser(description='Run the LCI framework')
        parser.add_argument('--config', type=str, default='config/lci_config.yaml',
                            help='Path to configuration file')
        parser.add_argument('--log-level', type=str, default='DEBUG',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Logging level')
        parser.add_argument('--output', type=str, 
                            help='Output directory (overrides config setting if provided)')
        args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level if hasattr(args, 'log_level') else 'INFO')
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if provided
    if hasattr(args, 'output') and args.output:
        config['output_dir'] = args.output
    
    # Run simulation
    output_dir = run_simulation(config)
    
    print(f"Simulation complete! Results are in {output_dir}")
    return output_dir

if __name__ == "__main__":
    main() 