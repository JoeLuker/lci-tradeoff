"""
LCI Framework - Main Script

This is the main entry point for running LCI framework experiments.
"""

import argparse
import logging
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

from lci_framework.environments.markov_environment import MarkovEnvironment
from lci_framework.agents.lci_agent import LCIAgent
from lci_framework.core.evolution import LCIEvolution
from lci_framework.utils.config import (
    setup_logging, 
    load_config, 
    create_default_config,
    default_logging_config,
    save_config
)
from lci_framework.utils.visualization import (
    plot_fitness_history,
    plot_lci_parameters,
    create_summary_dashboard
)
from lci_framework.utils.analysis import perform_comprehensive_analysis


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LCI Framework - Evolutionary experiments for L/C/I trade-offs")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run an LCI experiment")
    run_parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    run_parser.add_argument("--output-dir", "-o", type=str, default="results", 
                           help="Directory to save results")
    run_parser.add_argument("--name", "-n", type=str, default=None,
                           help="Experiment name (default: timestamp)")
    run_parser.add_argument("--verbose", "-v", action="store_true",
                           help="Enable verbose output")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize a new configuration file")
    init_parser.add_argument("--output", "-o", type=str, default="config.yaml",
                            help="Output path for the configuration file")
    init_parser.add_argument("--format", "-f", choices=["yaml", "json"], default="yaml",
                            help="Configuration file format")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze experiment results")
    analyze_parser.add_argument("--results-dir", "-r", type=str, required=True,
                               help="Directory containing experiment results")
    analyze_parser.add_argument("--output-dir", "-o", type=str, default="analysis",
                               help="Directory to save analysis results")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize experiment results")
    visualize_parser.add_argument("--results-dir", "-r", type=str, required=True,
                                 help="Directory containing experiment results")
    visualize_parser.add_argument("--output-dir", "-o", type=str, default="visualizations",
                                 help="Directory to save visualizations")
    visualize_parser.add_argument("--show", "-s", action="store_true",
                                 help="Show visualizations instead of saving them")
    
    return parser.parse_args()


def initialize_config(args):
    """Initialize a new configuration file"""
    config = create_default_config()
    
    output_path = args.output
    if not output_path.endswith((".yaml", ".yml", ".json")):
        if args.format == "yaml":
            output_path += ".yaml"
        else:
            output_path += ".json"
    
    try:
        # Create logging configuration
        log_config = default_logging_config()
        log_config_path = os.path.join(os.path.dirname(output_path), "logging_config.yaml")
        
        # Save configs
        save_config(config, output_path)
        save_config(log_config, log_config_path)
        
        print(f"Created configuration file: {output_path}")
        print(f"Created logging configuration: {log_config_path}")
        
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)


def run_experiment(args):
    """Run an LCI experiment with the specified configuration"""
    # Setup experiment name and output directory
    if args.name:
        experiment_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = os.path.join(output_dir, "experiment.log")
    setup_logging(default_level=log_level, log_file=log_file)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {experiment_name}")
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            # Save a copy of the configuration
            config_copy_path = os.path.join(output_dir, "config.yaml")
            save_config(config, config_copy_path)
        else:
            logger.info("No configuration file provided, using defaults")
            config = create_default_config()
            # Save the default configuration
            config_path = os.path.join(output_dir, "config.yaml")
            save_config(config, config_path)
        
        # Extract configuration parameters
        env_config = config.get("environment", {})
        evo_config = config.get("evolution", {})
        evo_config["output_dir"] = output_dir
        
        # Create and run evolution
        n_generations = evo_config.pop("n_generations", 50)
        steps_per_eval = evo_config.pop("steps_per_eval", 100)
        
        logger.info(f"Creating evolution with environment config: {env_config}")
        logger.info(f"Evolution config: {evo_config}")
        
        evolution = LCIEvolution(env_config=env_config, **evo_config)
        
        logger.info(f"Running simulation for {n_generations} generations")
        evolution.run_simulation(n_generations=n_generations, steps_per_eval=steps_per_eval)
        
        # Generate final visualizations
        logger.info("Generating final visualizations")
        evolution.plot_results(os.path.join(output_dir, "final_results.png"))
        
        # Find best agent
        logger.info("Evaluating final population")
        fitness_results = evolution.evaluate_fitness(steps_per_eval)
        best_result = max(fitness_results, key=lambda x: x["fitness"])
        best_agent = next(a for a in evolution.population if a.agent_id == best_result["agent_id"])
        
        # Save best agent
        best_agent_path = os.path.join(output_dir, "best_agent_final.pt")
        best_agent.save_model(best_agent_path)
        
        # Print summary statistics
        print("\n=== Experiment Complete ===")
        print(f"Results saved to: {output_dir}")
        print(f"Best agent fitness: {best_result['fitness']:.2f}")
        print(f"Best agent LCI balance: {best_result['lci_balance']:.3f}")
        print(f"Best agent parameters: {best_agent.lci_params}")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        print(f"Error running experiment: {e}")
        sys.exit(1)


def analyze_results(args):
    """Analyze experiment results"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Analyzing results from {args.results_dir}")
        analysis_results = perform_comprehensive_analysis(
            args.results_dir,
            args.output_dir
        )
        
        if "error" in analysis_results:
            logger.error(f"Analysis error: {analysis_results['error']}")
            print(f"Analysis error: {analysis_results['error']}")
            sys.exit(1)
            
        # Print summary
        print("\n=== Analysis Complete ===")
        print(f"Results saved to: {args.output_dir}")
        
        # Print key metrics if available
        if "fitness_analysis" in analysis_results:
            fitness = analysis_results["fitness_analysis"]
            if "overall_stats" in fitness and fitness["overall_stats"]["mean"] is not None:
                print(f"Average fitness: {fitness['overall_stats']['mean']:.2f}")
            if "improvement_rate" in fitness:
                print(f"Fitness improvement rate: {fitness['improvement_rate']:.4f}")
                
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}", exc_info=True)
        print(f"Error analyzing results: {e}")
        sys.exit(1)


def visualize_results(args):
    """Visualize experiment results"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Visualizing results from {args.results_dir}")
        
        # Find the most recent results file
        results_path = Path(args.results_dir)
        result_files = list(results_path.glob("stats_*.json"))
        
        if not result_files:
            logger.error("No result files found")
            print("No result files found")
            sys.exit(1)
            
        # Sort by modification time (most recent first)
        latest_file = sorted(result_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        logger.info(f"Using most recent results file: {latest_file}")
        
        # Load the results
        with open(latest_file, 'r') as f:
            results = json.load(f)
            
        # Create output directory if saving
        if not args.show:
            os.makedirs(args.output_dir, exist_ok=True)
            
        # Generate visualizations
        fitness_history = results.get("fitness_history", [])
        lci_history = results.get("lci_history", [])
        
        if fitness_history:
            save_path = os.path.join(args.output_dir, "fitness_history.png") if not args.show else None
            plot_fitness_history(fitness_history, save_path, args.show)
            
        if lci_history:
            save_path = os.path.join(args.output_dir, "lci_parameters.png") if not args.show else None
            plot_lci_parameters(lci_history, save_path, args.show)
            
        print("\n=== Visualization Complete ===")
        if not args.show:
            print(f"Visualizations saved to: {args.output_dir}")
            
        logger.info("Visualization completed successfully")
        
    except Exception as e:
        logger.error(f"Error visualizing results: {e}", exc_info=True)
        print(f"Error visualizing results: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.command == "init":
        initialize_config(args)
    elif args.command == "run":
        run_experiment(args)
    elif args.command == "analyze":
        analyze_results(args)
    elif args.command == "visualize":
        visualize_results(args)
    else:
        print("Please specify a command. Use --help for options.")
        sys.exit(1)


if __name__ == "__main__":
    main() 