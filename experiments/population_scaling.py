#!/usr/bin/env python3
"""
Population Scaling Experiment for LCI Framework

This script runs the LCI framework with various population sizes
to study how LCI strategies emerge at different population scales.
"""

import os
import yaml
import argparse
import logging
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from lci_framework.core.evolution import TensorEvolution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pop_scaling_experiment")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_experiment(config_path, pop_sizes, n_generations=3, steps_per_generation=25):
    """
    Run the scaling experiment with different population sizes.
    
    Args:
        config_path: Path to base configuration file
        pop_sizes: List of population sizes to test
        n_generations: Number of generations to run for each size
        steps_per_generation: Steps per generation
        
    Returns:
        Results DataFrame
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/pop_scaling_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Results storage
    results = {
        'pop_size': [],
        'run_time': [],
        'mean_fitness': [],
        'max_fitness': [],
        'mean_lci': [],
        'max_lci': []
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
    
    # Run for each population size
    for pop_size in pop_sizes:
        logger.info(f"Testing population size: {pop_size}")
        
        # Create experiment subdirectory
        experiment_output = f"{output_dir}/pop_{pop_size}"
        os.makedirs(experiment_output, exist_ok=True)
        
        # Update configuration
        config = base_config.copy()
        config['pop_size'] = pop_size
        config['n_generations'] = n_generations
        config['steps_per_generation'] = steps_per_generation
        config['output_dir'] = experiment_output
        
        # Initialize evolution
        evolution = TensorEvolution(
            pop_size=config.get('pop_size', 100),
            mutation_rate=config.get('mutation_rate', 0.1),
            tournament_size=config.get('tournament_size', 5),
            elitism=config.get('elitism', 0.1),
            n_states=config.get('n_states', 16),
            n_actions=config.get('n_actions', 4),
            n_generations=config.get('n_generations', 10),
            steps_per_generation=config.get('steps_per_generation', 25),
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
        results['pop_size'].append(pop_size)
        results['run_time'].append(run_time)
        results['mean_fitness'].append(final_gen['mean'])
        results['max_fitness'].append(final_gen['max'])
        results['mean_lci'].append(final_lci['mean'])
        results['max_lci'].append(final_lci['max'])
        
        logger.info(f"Population {pop_size} completed in {run_time:.2f} seconds")
        logger.info(f"Final mean fitness: {final_gen['mean']:.4f}, max: {final_gen['max']:.4f}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv = f"{output_dir}/scaling_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    # Plot scaling results
    plot_scaling_results(results_df, output_dir)
    
    return results_df, output_dir

def plot_scaling_results(results_df, output_dir):
    """
    Plot scaling experiment results.
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory for plots
    """
    # Plot runtime scaling
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['pop_size'], results_df['run_time'], 'o-', linewidth=2)
    plt.xlabel('Population Size')
    plt.ylabel('Run Time (seconds)')
    plt.title('Runtime Scaling with Population Size')
    plt.grid(True)
    plt.savefig(f"{output_dir}/runtime_scaling.png")
    
    # Plot fitness metrics scaling
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(results_df['pop_size'], results_df['mean_fitness'], 'o-', label='Mean Fitness')
    plt.plot(results_df['pop_size'], results_df['max_fitness'], 'o-', label='Max Fitness')
    plt.xlabel('Population Size')
    plt.ylabel('Fitness')
    plt.title('Fitness Scaling with Population Size')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(results_df['pop_size'], results_df['mean_lci'], 'o-', label='Mean LCI')
    plt.plot(results_df['pop_size'], results_df['max_lci'], 'o-', label='Max LCI')
    plt.xlabel('Population Size')
    plt.ylabel('LCI Value')
    plt.title('LCI Scaling with Population Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_scaling.png")

def main():
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(description='Run LCI population scaling experiment')
    parser.add_argument('--config', type=str, default='config/lci_config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--min-pop', type=int, default=100,
                        help='Minimum population size')
    parser.add_argument('--max-pop', type=int, default=1000,
                        help='Maximum population size')
    parser.add_argument('--step', type=int, default=100,
                        help='Population size step')
    parser.add_argument('--generations', type=int, default=3,
                        help='Number of generations per test')
    parser.add_argument('--steps', type=int, default=25,
                        help='Steps per generation')
    
    args = parser.parse_args()
    
    # Generate population sizes to test
    pop_sizes = list(range(args.min_pop, args.max_pop + 1, args.step))
    
    # Run the experiment
    results_df, output_dir = run_experiment(
        args.config, 
        pop_sizes, 
        n_generations=args.generations,
        steps_per_generation=args.steps
    )
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    # Print summary
    print("\nPopulation Scaling Results Summary:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main() 