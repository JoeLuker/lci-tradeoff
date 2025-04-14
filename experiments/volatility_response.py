#!/usr/bin/env python3
"""
Environment Volatility Response Experiment for LCI Framework

This script tests how LCI agents adapt to different environmental volatility patterns.
"""

import os
import yaml
import argparse
import logging
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from lci_framework.core.evolution import TensorEvolution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("volatility_experiment")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_volatility_experiment(config_path, volatility_patterns, n_generations=5, steps_per_generation=50):
    """
    Run the environment volatility experiment with different volatility patterns.
    
    Args:
        config_path: Path to base configuration file
        volatility_patterns: List of volatility pattern types to test
        n_generations: Number of generations to run for each pattern
        steps_per_generation: Steps per generation
        
    Returns:
        Results DataFrame
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/volatility_exp_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Results storage
    results = {
        'pattern': [],
        'run_time': [],
        'mean_fitness': [],
        'max_fitness': [],
        'mean_lci': [],
        'max_lci': [],
        'alive_count': []
    }
    
    # Load base config
    base_config = load_config(config_path)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available. Using CPU, which will be slower")
    
    # Define volatility pattern parameters
    volatility_params = {
        'stable': {
            'description': 'Stable environment with minimal changes',
            'env_update_freq': 0  # Never update
        },
        'gradual': {
            'description': 'Gradual environmental changes',
            'env_update_freq': 20  # Update every 20 steps
        },
        'cyclical': {
            'description': 'Cyclical environmental changes',
            'env_update_freq': 10,  # Update every 10 steps
            'cycle_pattern': True
        },
        'sudden': {
            'description': 'Sudden, dramatic environmental changes',
            'env_update_freq': 30,  # Update less frequently but more dramatically
            'shock_magnitude': 0.8
        },
        'random': {
            'description': 'Random, unpredictable environmental changes',
            'env_update_freq': 15,
            'randomize': True
        }
    }
    
    # Run for each volatility pattern
    for pattern in volatility_patterns:
        if pattern not in volatility_params:
            logger.warning(f"Unknown volatility pattern: {pattern}, skipping")
            continue
            
        logger.info(f"Testing volatility pattern: {pattern}")
        
        # Create experiment subdirectory
        experiment_output = f"{output_dir}/{pattern}"
        os.makedirs(experiment_output, exist_ok=True)
        
        # Update configuration
        config = base_config.copy()
        config['n_generations'] = n_generations
        config['steps_per_generation'] = steps_per_generation
        config['output_dir'] = experiment_output
        config['volatility'] = volatility_params[pattern]
        
        # Initialize evolution
        evolution = TensorEvolution(
            pop_size=config.get('pop_size', 100),
            mutation_rate=config.get('mutation_rate', 0.1),
            tournament_size=config.get('tournament_size', 5),
            elitism=config.get('elitism', 0.1),
            n_states=config.get('n_states', 16),
            n_actions=config.get('n_actions', 4),
            n_generations=config.get('n_generations', 5),
            steps_per_generation=config.get('steps_per_generation', 50),
            hidden_size=config.get('hidden_size', 64),
            n_layers=config.get('n_layers', 2),
            learning_rate=config.get('learning_rate', 0.001),
            dropout_rate=config.get('dropout_rate', 0.1),
            energy_cost_predict=config.get('energy_cost_predict', 0.01),
            energy_cost_learn=config.get('energy_cost_learn', 0.05),
            energy_init=config.get('energy_init', 10.0),
            energy_recovery=config.get('energy_recovery', 0.15),
            output_dir=experiment_output,
            device=device
        )
        
        # Configure environment volatility
        # Note: The actual configuration of environment volatility would need to be
        # implemented in the environment class, this is a placeholder for the idea
        if hasattr(evolution.env, 'set_volatility_pattern'):
            evolution.env.set_volatility_pattern(
                pattern_type=pattern,
                update_freq=volatility_params[pattern]['env_update_freq'],
                **{k: v for k, v in volatility_params[pattern].items() 
                   if k not in ['description', 'env_update_freq']}
            )
        else:
            logger.warning("Environment does not support volatility patterns. Results may not reflect volatility testing.")
        
        # Run and time the simulation
        start_time = time.time()
        sim_results = evolution.run_simulation()
        end_time = time.time()
        run_time = end_time - start_time
        
        # Save results
        evolution.save_results()
        evolution.plot_results()
        
        # Get the final generation stats
        final_gen = sim_results['fitness_history'][-1]
        final_lci = sim_results['lci_history'][-1]
        
        # Record results
        results['pattern'].append(pattern)
        results['run_time'].append(run_time)
        results['mean_fitness'].append(final_gen['mean'])
        results['max_fitness'].append(final_gen['max'])
        results['mean_lci'].append(final_lci['mean'])
        results['max_lci'].append(final_lci['max'])
        results['alive_count'].append(final_lci['alive_count'])
        
        logger.info(f"Pattern '{pattern}' completed in {run_time:.2f} seconds")
        logger.info(f"Final mean fitness: {final_gen['mean']:.4f}, max: {final_gen['max']:.4f}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv = f"{output_dir}/volatility_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    # Plot results
    plot_volatility_results(results_df, output_dir)
    
    return results_df, output_dir

def plot_volatility_results(results_df, output_dir):
    """
    Plot volatility experiment results.
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory for plots
    """
    patterns = results_df['pattern'].tolist()
    
    # Set up bar positions
    x = np.arange(len(patterns))
    width = 0.35
    
    # Fitness comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, results_df['mean_fitness'], width, label='Mean Fitness')
    plt.bar(x + width/2, results_df['max_fitness'], width, label='Max Fitness')
    plt.xlabel('Volatility Pattern')
    plt.ylabel('Fitness')
    plt.title('Impact of Environment Volatility on Fitness')
    plt.xticks(x, patterns)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # LCI comparison
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, results_df['mean_lci'], width, label='Mean LCI')
    plt.bar(x + width/2, results_df['max_lci'], width, label='Max LCI')
    plt.xlabel('Volatility Pattern')
    plt.ylabel('LCI Value')
    plt.title('Impact of Environment Volatility on LCI')
    plt.xticks(x, patterns)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/volatility_fitness_lci.png")
    
    # Alive agents and runtime comparison
    plt.figure(figsize=(12, 6))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Runtime line (primary y-axis)
    ax1.set_xlabel('Volatility Pattern')
    ax1.set_ylabel('Runtime (seconds)', color='tab:blue')
    ax1.plot(patterns, results_df['run_time'], 'o-', color='tab:blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Alive count (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Alive Agents', color='tab:red')
    ax2.plot(patterns, results_df['alive_count'], 's-', color='tab:red', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Runtime and Survival Rate by Volatility Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/volatility_runtime_survival.png")

def main():
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(description='Run LCI environment volatility experiment')
    parser.add_argument('--config', type=str, default='config/lci_config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--patterns', type=str, nargs='+', 
                        default=['stable', 'gradual', 'cyclical', 'sudden', 'random'],
                        help='Volatility patterns to test')
    parser.add_argument('--generations', type=int, default=5,
                        help='Number of generations per test')
    parser.add_argument('--steps', type=int, default=50,
                        help='Steps per generation')
    
    args = parser.parse_args()
    
    # Run the experiment
    results_df, output_dir = run_volatility_experiment(
        args.config, 
        args.patterns, 
        n_generations=args.generations,
        steps_per_generation=args.steps
    )
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    # Print summary
    print("\nVolatility Response Experiment Results Summary:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main() 