"""
Visualization Utilities

This module provides functions for visualizing LCI experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

def set_plot_style():
    """Set consistent aesthetic style for plots"""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12


def plot_fitness_history(
    fitness_history: List[float], 
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot fitness history over generations
    
    Args:
        fitness_history: List of average fitness values per generation
        save_path: Path to save the plot, or None to not save
        show: Whether to display the plot
    """
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    plt.plot(fitness_history)
    plt.title('Average Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    
    if save_path:
        try:
            plt.savefig(save_path)
            logger.info(f"Saved fitness history plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_lci_parameters(
    lci_history: List[Tuple[float, float, float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot LCI parameter values over generations
    
    Args:
        lci_history: List of (L, C, I) tuples per generation
        save_path: Path to save the plot, or None to not save
        show: Whether to display the plot
    """
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    L_values, C_values, I_values = zip(*lci_history)
    
    plt.plot(L_values, 'r-', label='L (Losslessness)')
    plt.plot(C_values, 'g-', label='C (Compression)')
    plt.plot(I_values, 'b-', label='I (Invariance)')
    
    plt.title('LCI Parameters Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        try:
            plt.savefig(save_path)
            logger.info(f"Saved LCI parameters plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_lci_space(
    agents_lci: List[Dict[str, float]],
    fitness_values: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot agents in LCI space
    
    Args:
        agents_lci: List of dictionaries with L, C, I values
        fitness_values: Optional list of fitness values for color coding
        save_path: Path to save the plot, or None to not save
        show: Whether to display the plot
    """
    set_plot_style()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract L, C, I values
    L = [agent["L"] for agent in agents_lci]
    C = [agent["C"] for agent in agents_lci]
    I = [agent["I"] for agent in agents_lci]
    
    # Plot points
    if fitness_values:
        # Normalize fitness for color mapping
        norm_fitness = np.array(fitness_values)
        if np.max(norm_fitness) > np.min(norm_fitness):
            norm_fitness = (norm_fitness - np.min(norm_fitness)) / (np.max(norm_fitness) - np.min(norm_fitness))
        
        scatter = ax.scatter(L, C, I, c=norm_fitness, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Normalized Fitness')
    else:
        ax.scatter(L, C, I, c='blue', s=50, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('L (Losslessness)')
    ax.set_ylabel('C (Compression)')
    ax.set_zlabel('I (Invariance)')
    plt.title('Agents in LCI Parameter Space')
    
    # Add the "balanced" line where L=C=I
    if max(max(L), max(C), max(I)) > 0:
        line_max = max(max(L), max(C), max(I))
        line_min = min(min(L), min(C), min(I))
        balanced_line = np.linspace(line_min, line_max, 100)
        ax.plot(balanced_line, balanced_line, balanced_line, 'r-', label='L=C=I (Optimal Balance)')
        plt.legend()
    
    if save_path:
        try:
            plt.savefig(save_path)
            logger.info(f"Saved LCI space plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_lci_balance_distribution(
    balance_values: List[float],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot distribution of LCI balance values
    
    Args:
        balance_values: List of LCI balance values
        save_path: Path to save the plot, or None to not save
        show: Whether to display the plot
    """
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    sns.histplot(balance_values, bins=20, kde=True)
    plt.axvline(np.mean(balance_values), color='r', linestyle='--', label=f'Mean: {np.mean(balance_values):.3f}')
    
    plt.title('Distribution of LCI Balance Values')
    plt.xlabel('LCI Balance Score')
    plt.ylabel('Count')
    plt.legend()
    
    if save_path:
        try:
            plt.savefig(save_path)
            logger.info(f"Saved LCI balance distribution plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_environment_volatility(
    volatility_history: List[float],
    performance_history: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot environment volatility over time
    
    Args:
        volatility_history: List of volatility values
        performance_history: Optional list of agent performance metrics at each step
        save_path: Path to save the plot, or None to not save
        show: Whether to display the plot
    """
    set_plot_style()
    plt.figure(figsize=(12, 6))
    
    time_steps = list(range(len(volatility_history)))
    
    # Plot volatility
    plt.plot(time_steps, volatility_history, 'b-', label='Environment Volatility')
    
    # If performance history is provided, plot on secondary axis
    if performance_history and len(performance_history) == len(volatility_history):
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(time_steps, performance_history, 'r-', label='Agent Performance')
        ax2.set_ylabel('Performance', color='r')
        ax2.tick_params(axis='y', colors='r')
        
        # Add both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        plt.legend()
    
    plt.title('Environment Volatility Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Volatility Level')
    plt.grid(True)
    
    if save_path:
        try:
            plt.savefig(save_path)
            logger.info(f"Saved volatility plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_dashboard(
    fitness_history: List[float],
    lci_history: List[Tuple[float, float, float]],
    agents: List[Any],
    best_agent: Any,
    output_dir: str,
    identifier: str = "summary"
):
    """
    Create a comprehensive dashboard of visualization results
    
    Args:
        fitness_history: List of fitness values per generation
        lci_history: List of (L, C, I) tuples per generation
        agents: List of agent objects
        best_agent: Best performing agent
        output_dir: Directory to save outputs
        identifier: Identifier string for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a multi-panel figure
    set_plot_style()
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Fitness History
    plt.subplot(2, 2, 1)
    plt.plot(fitness_history)
    plt.title('Average Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    
    # Plot 2: LCI Parameters Over Time
    plt.subplot(2, 2, 2)
    L_values, C_values, I_values = zip(*lci_history)
    plt.plot(L_values, 'r-', label='L')
    plt.plot(C_values, 'g-', label='C')
    plt.plot(I_values, 'b-', label='I')
    plt.title('LCI Parameters Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot 3: Final Population in L-C Space (with I as color)
    plt.subplot(2, 2, 3)
    L = [agent.lci_params["L"] for agent in agents]
    C = [agent.lci_params["C"] for agent in agents]
    I = [agent.lci_params["I"] for agent in agents]
    
    scatter = plt.scatter(L, C, c=I, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='I Value')
    plt.title('Final Population in L-C Space')
    plt.xlabel('L (Losslessness)')
    plt.ylabel('C (Compression)')
    
    # Highlight best agent
    best_L = best_agent.lci_params["L"]
    best_C = best_agent.lci_params["C"]
    plt.scatter([best_L], [best_C], c='red', s=100, marker='*', label='Best Agent')
    plt.legend()
    
    # Plot 4: LCI Balance Distribution
    plt.subplot(2, 2, 4)
    balance_values = [agent.get_lci_balance() for agent in agents]
    plt.hist(balance_values, bins=20)
    plt.axvline(best_agent.get_lci_balance(), color='r', linestyle='--', 
                label=f'Best Agent: {best_agent.get_lci_balance():.3f}')
    plt.title('LCI Balance Distribution')
    plt.xlabel('LCI Balance')
    plt.ylabel('Count')
    plt.legend()
    
    # Save the dashboard
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, f"dashboard_{identifier}.png")
    plt.savefig(dashboard_path)
    plt.close()
    
    logger.info(f"Created summary dashboard: {dashboard_path}")
    
    return dashboard_path 