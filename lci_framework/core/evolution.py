"""
GPU-Optimized Tensor Evolution

This module implements a fully vectorized evolutionary algorithm that 
optimizes agent populations using GPU acceleration.
"""

import torch
import numpy as np
import os
import logging
import time
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from tqdm import tqdm

from lci_framework.agents.agent import TensorLCIAgent
from lci_framework.environments.environment import VectorizedMarkovEnvironment

# Configure logger
logger = logging.getLogger(__name__)

class TensorEvolution:
    """
    GPU-accelerated evolution framework for optimizing LCI agent populations.
    
    Uses tensor operations to efficiently evaluate and evolve agent populations.
    """
    
    def __init__(self,
                pop_size: int = 100,
                mutation_rate: float = 0.1,
                tournament_size: int = 5,
                elitism: int = 1,
                n_states: int = 10,
                n_actions: int = 4,
                n_generations: int = 100,
                steps_per_generation: int = 1000,
                hidden_size: int = 64,
                n_layers: int = 2,
                learning_rate: float = 0.01,
                dropout_rate: float = 0.1,
                l1_reg: float = 0.001,
                l2_reg: float = 0.001,
                energy_cost_predict: float = 0.01,
                energy_cost_learn: float = 0.1,
                energy_init: float = 1.0,
                energy_recovery: float = 0.05,
                output_dir: str = "results",
                device: Optional[torch.device] = None):
        """
        Initialize the tensor evolution framework.
        
        Args:
            pop_size: Number of agents in the population
            mutation_rate: Probability of mutation for each parameter
            tournament_size: Number of agents per tournament
            elitism: Number of top agents to preserve each generation
            n_states: Number of states in the environment
            n_actions: Number of actions in the environment
            n_generations: Number of generations to evolve
            steps_per_generation: Number of environment steps for each generation
            hidden_size: Size of hidden layers in neural networks
            n_layers: Number of layers in neural networks
            learning_rate: Learning rate for neural network training
            dropout_rate: Dropout rate for neural networks
            l1_reg: L1 regularization weight
            l2_reg: L2 regularization weight
            energy_cost_predict: Energy cost for making a prediction
            energy_cost_learn: Energy cost for learning
            energy_init: Initial energy level
            energy_recovery: Energy recovery rate per step
            output_dir: Directory to save results
            device: Device for computation (auto-detected if None)
        """
        # Validate parameters
        assert pop_size > 0, "Population size must be positive"
        assert 0 <= mutation_rate <= 1, "Mutation rate must be between 0 and 1"
        assert 1 <= tournament_size <= pop_size, "Tournament size must be between 1 and population size"
        assert 0 <= elitism <= pop_size, "Elitism must be between 0 and population size"
        assert n_generations > 0, "Number of generations must be positive"
        assert steps_per_generation > 0, "Steps per generation must be positive"
        
        # Store parameters
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.n_generations = n_generations
        self.steps_per_generation = steps_per_generation
        self.output_dir = output_dir
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Set up device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS device for evolution")
                print("Using MPS device (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA device for evolution")
                print("Using CUDA device (NVIDIA GPU)")
            else:
                raise RuntimeError("No GPU available. This implementation requires GPU acceleration.")
        else:
            self.device = device
            logger.info(f"Using specified device: {device}")
        
        # Create vectorized environment
        self.env = VectorizedMarkovEnvironment(
            n_states=n_states,
            n_actions=n_actions,
            device=self.device
        )
        
        # Initialize environment with population size
        self.env.reset(pop_size=pop_size)
        
        # Create population of agents
        self.agents = TensorLCIAgent(
            pop_size=pop_size,
            input_size=n_states,
            energy_cost_predict=energy_cost_predict,
            energy_cost_learn=energy_cost_learn,
            energy_init=energy_init,
            energy_recovery=energy_recovery,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            device=self.device
        )
        
        # Create directory for results
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.generation = 0
        self.fitness_history = []
        self.lci_history = []
        self.best_agent_history = []
        self.best_fitness = float('-inf')
        self.best_agent_generation = 0
        
        logger.info(f"Tensor Evolution initialized with {pop_size} agents")
    
    def evaluate_fitness(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the fitness of the entire population of agents using the environment.
        
        Returns:
            Tuple of (fitness_tensor, lci_tensor)
        """
        # Reset environment for evaluation
        states = self.env.reset(pop_size=self.pop_size)
        
        # Initialize fitness tensor for all agents
        fitness = torch.zeros(self.pop_size, device=self.device)
        lci_values = torch.zeros(self.pop_size, device=self.device)
        
        # Get alive mask for all agents
        alive_mask = self.agents.get_energy() > 0
        
        # Run simulation for steps_per_generation steps
        total_rewards = torch.zeros(self.pop_size, device=self.device)
        step_counts = torch.zeros(self.pop_size, device=self.device)
        
        for _ in tqdm(range(self.steps_per_generation), desc=f"Generation {self.generation+1}"):
            # Get predictions for all agents
            predictions = self.agents.predict(states)
            
            # Step environment forward
            next_states, rewards, dones, _ = self.env.step_batch(predictions)
            
            # Update total rewards and step counts for alive agents
            for i in range(self.pop_size):
                if alive_mask[i]:
                    total_rewards[i] += rewards[i]
                    step_counts[i] += 1
            
            # Learn from transition for all alive agents
            self.agents.learn(states, rewards)
            
            # Update states
            states = next_states
            
            # Update energy levels
            self.agents.update_energy()
            
            # Update alive mask
            alive_mask = self.agents.get_energy() > 0
            
            # If all agents are dead, break early
            if not alive_mask.any():
                logger.warning("All agents depleted energy, ending generation early")
                break
        
        # Calculate fitness as total reward
        fitness = total_rewards
        
        # Calculate LCI as average reward per step (for agents that took at least one step)
        for i in range(self.pop_size):
            if step_counts[i] > 0:
                lci_values[i] = total_rewards[i] / step_counts[i]
        
        return fitness, lci_values

    def selection_and_reproduction(self, fitness: torch.Tensor) -> None:
        """
        Select parents based on tournament selection and create new population.
        
        Args:
            fitness: Fitness tensor for current population
        """
        # Ensure fitness is on the correct device
        fitness = fitness.to(self.device)
        
        # Initialize new population
        new_population = TensorLCIAgent(
            pop_size=self.pop_size,
            input_size=self.n_states,
            energy_cost_predict=self.agents.energy_cost_predict,
            energy_cost_learn=self.agents.energy_cost_learn,
            energy_init=self.agents.energy_init,
            energy_recovery=self.agents.energy_recovery,
            learning_rate=self.agents.learning_rate,
            hidden_size=self.agents.model.hidden_size,
            n_layers=self.agents.model.n_layers,
            dropout_rate=self.agents.model.dropout_rate,
            l1_reg=self.agents.l1_reg,
            l2_reg=self.agents.l2_reg,
            device=self.device
        )
        
        # Elitism: Copy the best agents
        num_elite = max(1, int(self.elitism * self.pop_size) if isinstance(self.elitism, float) else self.elitism)
        
        if num_elite > 0:
            # Get indices of elite agents
            elite_indices = torch.topk(fitness, num_elite).indices
            
            # Copy elite agents' parameters - using detach() to avoid gradient issues
            for i, idx in enumerate(elite_indices):
                # Copy model parameters (this would need model-specific logic)
                for target_layer, source_layer in zip(new_population.model.layers, self.agents.model.layers):
                    # Using assign instead of in-place operations to avoid autograd issues
                    weight_copy = source_layer.weight[idx].clone().detach()
                    with torch.no_grad():
                        target_layer.weight[i].copy_(weight_copy)
                    
                    if target_layer.bias is not None:
                        bias_copy = source_layer.bias[idx].clone().detach()
                        with torch.no_grad():
                            target_layer.bias[i].copy_(bias_copy)
                
                # Copy output layer parameters
                output_weight_copy = self.agents.model.output_layer.weight[idx].clone().detach()
                with torch.no_grad():
                    new_population.model.output_layer.weight[i].copy_(output_weight_copy)
                
                if new_population.model.output_layer.bias is not None:
                    output_bias_copy = self.agents.model.output_layer.bias[idx].clone().detach()
                    with torch.no_grad():
                        new_population.model.output_layer.bias[i].copy_(output_bias_copy)
        
        # Tournament selection for the rest of the population
        for i in range(num_elite, self.pop_size):
            # Select tournament participants
            tournament_indices = torch.randint(0, self.pop_size, (self.tournament_size,), device=self.device)
            tournament_fitness = fitness[tournament_indices]
            
            # Select the winner
            winner_idx = tournament_indices[torch.argmax(tournament_fitness)]
            
            # Copy winner's parameters to new population
            for target_layer, source_layer in zip(new_population.model.layers, self.agents.model.layers):
                weight_copy = source_layer.weight[winner_idx].clone().detach()
                with torch.no_grad():
                    target_layer.weight[i].copy_(weight_copy)
                
                if target_layer.bias is not None:
                    bias_copy = source_layer.bias[winner_idx].clone().detach()
                    with torch.no_grad():
                        target_layer.bias[i].copy_(bias_copy)
                    
            # Copy output layer parameters
            output_weight_copy = self.agents.model.output_layer.weight[winner_idx].clone().detach()
            with torch.no_grad():
                new_population.model.output_layer.weight[i].copy_(output_weight_copy)
            
            if new_population.model.output_layer.bias is not None:
                output_bias_copy = self.agents.model.output_layer.bias[winner_idx].clone().detach()
                with torch.no_grad():
                    new_population.model.output_layer.bias[i].copy_(output_bias_copy)
        
        # Apply mutation to non-elite agents
        for i in range(num_elite, self.pop_size):
            # Apply mutation with probability mutation_rate
            mutation_mask = torch.rand(size=(1,), device=self.device) < self.mutation_rate
            
            if mutation_mask.item():
                # Mutate parameters
                for layer in new_population.model.layers:
                    # Add Gaussian noise to weights
                    with torch.no_grad():
                        layer.weight[i] += torch.randn_like(layer.weight[i]) * 0.1
                    
                    if layer.bias is not None:
                        with torch.no_grad():
                            layer.bias[i] += torch.randn_like(layer.bias[i]) * 0.1
                
                # Mutate output layer
                with torch.no_grad():
                    new_population.model.output_layer.weight[i] += torch.randn_like(new_population.model.output_layer.weight[i]) * 0.1
                
                if new_population.model.output_layer.bias is not None:
                    with torch.no_grad():
                        new_population.model.output_layer.bias[i] += torch.randn_like(new_population.model.output_layer.bias[i]) * 0.1
        
        # Replace the old population with the new one
        self.agents = new_population
        
        # Reset environment for the new generation
        self.env.update_environment()
        # Initialize environment with the population size
        self.env.reset(pop_size=self.pop_size)
    
    def run_generation(self, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single generation of the evolutionary algorithm.
        
        Args:
            steps: Number of steps to run
            
        Returns:
            Tuple of (fitness_tensor, lci_tensor)
        """
        # Store the current steps_per_generation
        original_steps = self.steps_per_generation
        self.steps_per_generation = steps
        
        # Evaluate fitness
        fitness, lci_values = self.evaluate_fitness()
        
        # Track the best agent
        best_idx = torch.argmax(fitness).item()
        best_fitness_gen = fitness[best_idx].item()
        
        if best_fitness_gen > self.best_fitness:
            self.best_fitness = best_fitness_gen
            self.best_agent_generation = self.generation
            
            # Save the best agent
            os.makedirs(f"{self.output_dir}/best_agents", exist_ok=True)
            self.agents.save_agent(f"{self.output_dir}/best_agents/best_agent_gen_{self.generation}")
        
        # Log statistics
        mean_fitness = fitness.mean().item()
        std_fitness = fitness.std().item()
        max_fitness = fitness.max().item()
        mean_lci = lci_values.mean().item()
        alive_count = (self.agents.get_energy() > 0).sum().item()
        
        logger.info(f"Generation {self.generation+1}: "
                   f"Mean Fitness = {mean_fitness:.4f}, "
                   f"Max Fitness = {max_fitness:.4f}, "
                   f"Mean LCI = {mean_lci:.4f}, "
                   f"Alive Agents = {alive_count}/{self.pop_size}")
        
        # Store history
        self.fitness_history.append({
            'mean': mean_fitness,
            'std': std_fitness,
            'max': max_fitness
        })
        self.lci_history.append({
            'mean': mean_lci,
            'max': lci_values.max().item(),
            'alive_count': alive_count
        })
        self.best_agent_history.append({
            'generation': self.generation,
            'fitness': best_fitness_gen,
            'lci': lci_values[best_idx].item()
        })
        
        # Increment generation counter
        self.generation += 1
        
        # Selection and reproduction
        self.selection_and_reproduction(fitness)
        
        # Restore original steps
        self.steps_per_generation = original_steps
        
        return fitness, lci_values
    
    def run_simulation(self, n_generations=None, steps_per_generation=None, learning_rate=None) -> Dict:
        """
        Run the evolutionary simulation for n_generations.
        
        Args:
            n_generations: Override the number of generations to run
            steps_per_generation: Override the number of steps per generation
            learning_rate: Override the learning rate for the agents
            
        Returns:
            Dictionary of results
        """
        start_time = time.time()
        
        # Update parameters if provided
        if n_generations is not None:
            self.n_generations = n_generations
        if steps_per_generation is not None:
            self.steps_per_generation = steps_per_generation
        if learning_rate is not None:
            self.agents.learning_rate = learning_rate
        
        # Reset generation counter
        self.generation = 0
        
        # Clear history
        self.fitness_history = []
        self.lci_history = []
        self.best_agent_history = []
        self.best_fitness = float('-inf')
        self.best_agent_generation = 0
        
        # Run for n_generations
        for gen in range(self.n_generations):
            # Run a single generation
            self.run_generation(self.steps_per_generation)
            
            # Save current results periodically
            if gen % 10 == 0 or gen == self.n_generations - 1:
                self.save_results()
                self.plot_results()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best agent found in generation {self.best_agent_generation} with fitness {self.best_fitness:.4f}")
        
        # Save final results
        self.save_results()
        self.plot_results()
        
        return {
            'fitness_history': self.fitness_history,
            'lci_history': self.lci_history,
            'best_agent': {
                'generation': self.best_agent_generation,
                'fitness': self.best_fitness
            },
            'elapsed_time': elapsed_time
        }
    
    def plot_results(self) -> None:
        """
        Plot the fitness and LCI history.
        """
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Extract data for plotting
        generations = list(range(1, len(self.fitness_history) + 1))
        mean_fitness = [gen['mean'] for gen in self.fitness_history]
        max_fitness = [gen['max'] for gen in self.fitness_history]
        mean_lci = [gen['mean'] for gen in self.lci_history]
        max_lci = [gen['max'] for gen in self.lci_history]
        alive_counts = [gen['alive_count'] for gen in self.lci_history]
        
        # Plot fitness history
        ax1.plot(generations, mean_fitness, label='Mean Fitness', color='blue')
        ax1.plot(generations, max_fitness, label='Max Fitness', color='green')
        ax1.set_title('Fitness History')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.legend()
        ax1.grid(True)
        
        # Plot LCI history
        ax2.plot(generations, mean_lci, label='Mean LCI', color='blue')
        ax2.plot(generations, max_lci, label='Max LCI', color='green')
        
        # Create a second y-axis for alive counts
        ax2_alive = ax2.twinx()
        ax2_alive.plot(generations, alive_counts, label='Alive Agents', color='red', linestyle='--')
        ax2_alive.set_ylabel('Alive Agents Count', color='red')
        
        ax2.set_title('LCI History and Alive Agents')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('LCI')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_alive.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)
        
        ax2.grid(True)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/evolution_history.png")
        plt.close()
    
    def save_results(self) -> None:
        """
        Save the results to a JSON file.
        """
        results = {
            'parameters': {
                'pop_size': self.pop_size,
                'mutation_rate': self.mutation_rate,
                'tournament_size': self.tournament_size,
                'elitism': self.elitism,
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'n_generations': self.n_generations,
                'steps_per_generation': self.steps_per_generation
            },
            'fitness_history': self.fitness_history,
            'lci_history': self.lci_history,
            'best_agent_history': self.best_agent_history
        }
        
        with open(f"{self.output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2) 