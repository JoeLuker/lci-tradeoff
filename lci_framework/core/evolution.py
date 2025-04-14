"""
Evolution Framework Module

This module implements a population-based evolutionary framework for optimizing
LCI (Losslessness, Compression, Invariance) agent parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from pathlib import Path
import json
import os
from datetime import datetime
from tqdm import tqdm

from lci_framework.environments.markov_environment import MarkovEnvironment
from lci_framework.agents.lci_agent import LCIAgent

# Configure logger
logger = logging.getLogger(__name__)

class LCIEvolution:
    """
    Population-based optimization for LCI strategies
    
    This class manages a population of agents with different LCI parameters,
    evaluates their fitness, and evolves the population over time.
    
    Attributes:
        env_config (dict): Configuration for the environment
        pop_size (int): Population size
        mutation_rate (float): Probability of mutation during reproduction
        tournament_size (int): Number of agents to select for tournament selection
        elitism (int): Number of top agents to preserve as-is
        env (MarkovEnvironment): The environment for agent evaluation
        population (list): List of LCIAgent instances
        generation (int): Current generation number
        fitness_history (list): History of average fitness values
        lci_history (list): History of average LCI parameters
    """
    
    def __init__(
        self,
        env_config: Dict,
        pop_size: int = 50,
        mutation_rate: float = 0.1,
        tournament_size: int = 5,
        elitism: int = 2,
        seed: Optional[int] = None,
        output_dir: str = "results"
    ):
        """
        Initialize the evolution framework
        
        Args:
            env_config: Configuration for the environment
            pop_size: Population size
            mutation_rate: Probability of mutation during reproduction
            tournament_size: Number of agents to select for tournament selection
            elitism: Number of top agents to preserve as-is
            seed: Random seed for reproducibility
            output_dir: Directory for saving results
        """
        # Input validation
        if pop_size <= 0:
            raise ValueError("Population size must be positive")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if tournament_size <= 0 or tournament_size > pop_size:
            raise ValueError("Tournament size must be positive and not greater than population size")
        if elitism < 0 or elitism > pop_size // 2:
            raise ValueError("Elitism must be non-negative and not greater than half the population size")
            
        self.env_config = env_config
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"Set random seed: {seed}")
        
        # Create environment
        self.env = MarkovEnvironment(**env_config)
        
        # Initialize population
        self.population = []
        self._initialize_population()
        
        # Tracking statistics
        self.generation = 0
        self.fitness_history = []
        self.lci_history = []
        self.best_agent_history = []
        
        logger.info(f"Initialized evolution with population size {pop_size}")
        
    def _initialize_population(self):
        """
        Initialize population with random LCI parameters
        
        Creates a diverse initial population with different LCI parameter combinations.
        """
        for i in range(self.pop_size):
            # Randomize LCI parameters for each agent
            agent = LCIAgent(
                input_size=self.env.n_states,
                learning_rate=random.uniform(0.001, 0.1),   # L parameter
                hidden_size=random.randint(8, 128),         # C parameter
                n_layers=random.randint(1, 3),              # C parameter
                l1_reg=random.uniform(0, 0.01),            # I parameter
                l2_reg=random.uniform(0, 0.01),            # I parameter
                dropout_rate=random.uniform(0, 0.5),       # I parameter
                agent_id=i
            )
            self.population.append(agent)
            
        logger.debug(f"Created initial population of {len(self.population)} agents")
    
    def evaluate_fitness(self, steps_per_eval: int = 100) -> List[Dict]:
        """
        Evaluate fitness of all agents in the population
        
        Args:
            steps_per_eval: Number of steps to run each evaluation
            
        Returns:
            List of fitness results for each agent
        """
        results = []
        
        # Check if multiprocessing should be used
        use_multiprocessing = self.pop_size > 4
        
        if use_multiprocessing:
            # Process agents efficiently using multiprocessing
            num_processes = min(mp.cpu_count(), self.pop_size)
            logger.info(f"Evaluating population using {num_processes} processes")
            
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(
                    self._evaluate_agent_fitness,
                    [(agent, steps_per_eval) for agent in self.population]
                )
        else:
            # Sequential processing for debugging or small populations
            logger.info("Evaluating population sequentially")
            for agent in tqdm(self.population, desc="Evaluating agents"):
                results.append(self._evaluate_agent_fitness((agent, steps_per_eval)))
        
        # Update fitness history
        avg_fitness = np.mean([r["fitness"] for r in results])
        self.fitness_history.append(avg_fitness)
        
        # Update LCI history
        l_avg = np.mean([agent.lci_params["L"] for agent in self.population])
        c_avg = np.mean([agent.lci_params["C"] for agent in self.population])
        i_avg = np.mean([agent.lci_params["I"] for agent in self.population])
        
        self.lci_history.append((l_avg, c_avg, i_avg))
        
        # Track best agent
        best_result = max(results, key=lambda x: x["fitness"])
        self.best_agent_history.append(best_result)
        
        return results
    
    def _evaluate_agent_fitness(self, args) -> Dict:
        """
        Helper function for parallel fitness evaluation
        
        Args:
            args: Tuple of (agent, steps_per_eval)
            
        Returns:
            Dictionary with fitness results
        """
        agent, steps_per_eval = args
        
        # Create a separate environment for this evaluation
        env = MarkovEnvironment(**self.env_config)
        
        # Initialize trackers
        total_reward = 0
        prediction_errors = []
        survived_steps = 0
        
        # Reset environment
        obs = env.get_observation(env.reset())
        
        # Run simulation for specified number of steps
        for t in range(steps_per_eval):
            # Update environment volatility
            env.update_environment(t)
            
            # Agent predicts next state
            pred_state = agent.predict(obs)
            
            # Environment takes a step
            next_state, reward = env.step()
            next_obs = env.get_observation(next_state)
            
            # Agent learns from transition
            error = agent.learn(obs, next_obs)
            prediction_errors.append(error)
            
            # Agent receives reward
            agent.receive_reward(reward)
            total_reward += reward
            
            # Check if agent is still alive
            if not agent.is_alive():
                break
                
            # Update observation
            obs = next_obs
            survived_steps = t + 1
        
        # Calculate fitness
        fitness = total_reward
        
        # Calculate stability across environment changes
        # Higher is better - less variation in performance during volatility changes
        if len(prediction_errors) > 1:
            stability = 1.0 / (1.0 + np.std(prediction_errors))
        else:
            stability = 0
            
        # Calculate efficiency
        efficiency = agent.energy / agent.initial_energy
        
        # Calculate LCI balance
        lci_balance = agent.get_lci_balance()
        
        return {
            "agent_id": agent.agent_id,
            "fitness": fitness,
            "reward": total_reward,
            "steps_survived": survived_steps,
            "stability": stability,
            "efficiency": efficiency,
            "lci_balance": lci_balance,
            "lci_params": agent.lci_params,
            "final_energy": agent.energy
        }
    
    def selection_and_reproduction(self, fitness_results: List[Dict]):
        """
        Select parents and create next generation
        
        Args:
            fitness_results: List of fitness evaluation results
        """
        # Sort agents by fitness
        sorted_results = sorted(fitness_results, key=lambda x: x["fitness"], reverse=True)
        
        # Create next generation
        next_generation = []
        
        # Elitism: Keep top performing agents
        for i in range(self.elitism):
            if i < len(sorted_results):
                elite_id = sorted_results[i]["agent_id"]
                elite_agent = next(a for a in self.population if a.agent_id == elite_id)
                next_generation.append(elite_agent)
                logger.debug(f"Elite agent {elite_id} preserved with fitness {sorted_results[i]['fitness']:.3f}")
        
        # Fill the rest with offspring from tournament selection
        while len(next_generation) < self.pop_size:
            # Tournament selection for parent 1
            parent1 = self._tournament_selection(fitness_results)
            
            # Tournament selection for parent 2
            parent2 = self._tournament_selection(fitness_results)
            
            # Create offspring through crossover and mutation
            offspring = self._crossover_and_mutate(parent1, parent2)
            next_generation.append(offspring)
        
        # Update population
        self.population = next_generation
        self.generation += 1
        
        logger.info(f"Created generation {self.generation} with {len(self.population)} agents")
    
    def _tournament_selection(self, fitness_results: List[Dict]) -> LCIAgent:
        """
        Select an agent using tournament selection
        
        Args:
            fitness_results: List of fitness evaluation results
            
        Returns:
            Selected agent
        """
        # Randomly select tournament_size agents
        tournament_indices = random.sample(range(len(fitness_results)), min(self.tournament_size, len(fitness_results)))
        tournament = [fitness_results[i] for i in tournament_indices]
        
        # Select the one with highest fitness
        winner_id = max(tournament, key=lambda x: x["fitness"])["agent_id"]
        
        # Find and return the corresponding agent
        winner = next(a for a in self.population if a.agent_id == winner_id)
        return winner
    
    def _crossover_and_mutate(self, parent1: LCIAgent, parent2: LCIAgent) -> LCIAgent:
        """
        Create offspring through crossover and mutation
        
        Args:
            parent1: First parent agent
            parent2: Second parent agent
            
        Returns:
            New offspring agent
        """
        # Crossover LCI parameters
        # For each parameter, randomly choose from either parent
        learning_rate = random.choice([parent1.learning_rate, parent2.learning_rate])
        hidden_size = random.choice([parent1.hidden_size, parent2.hidden_size])
        n_layers = random.choice([parent1.n_layers, parent2.n_layers])
        l1_reg = random.choice([parent1.l1_reg, parent2.l1_reg])
        l2_reg = random.choice([parent1.l2_reg, parent2.l2_reg])
        dropout_rate = random.choice([parent1.dropout_rate, parent2.dropout_rate])
        
        # Apply mutation with probability mutation_rate
        if random.random() < self.mutation_rate:
            # Mutate learning rate (L parameter)
            learning_rate = max(0.001, min(0.1, learning_rate * random.uniform(0.5, 1.5)))
            
        if random.random() < self.mutation_rate:
            # Mutate hidden size (C parameter)
            hidden_size = max(4, min(256, int(hidden_size * random.uniform(0.5, 1.5))))
            
        if random.random() < self.mutation_rate:
            # Mutate number of layers (C parameter)
            n_layers = max(1, min(4, n_layers + random.choice([-1, 0, 1])))
            
        if random.random() < self.mutation_rate:
            # Mutate L1 regularization (I parameter)
            l1_reg = max(0, min(0.05, l1_reg * random.uniform(0.5, 1.5)))
            
        if random.random() < self.mutation_rate:
            # Mutate L2 regularization (I parameter)
            l2_reg = max(0, min(0.05, l2_reg * random.uniform(0.5, 1.5)))
            
        if random.random() < self.mutation_rate:
            # Mutate dropout rate (I parameter)
            dropout_rate = max(0, min(0.8, dropout_rate * random.uniform(0.5, 1.5)))
        
        # Create new agent with crossed over and mutated parameters
        offspring_id = self.generation * self.pop_size + len(self.population)
        
        return LCIAgent(
            input_size=parent1.input_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            n_layers=n_layers,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            agent_id=offspring_id
        )
    
    def run_simulation(self, n_generations: int = 50, steps_per_eval: int = 100):
        """
        Run the full evolutionary simulation
        
        Args:
            n_generations: Number of generations to simulate
            steps_per_eval: Number of steps per fitness evaluation
        """
        simulation_start_time = datetime.now()
        logger.info(f"Starting simulation with {n_generations} generations")
        
        for gen in tqdm(range(n_generations), desc="Generation"):
            # Evaluate fitness
            fitness_results = self.evaluate_fitness(steps_per_eval)
            
            # Calculate statistics
            avg_fitness = np.mean([r["fitness"] for r in fitness_results])
            max_fitness = np.max([r["fitness"] for r in fitness_results])
            avg_lci_balance = np.mean([r["lci_balance"] for r in fitness_results])
            
            logger.info(f"Generation {gen}: Avg Fitness = {avg_fitness:.2f}, Max Fitness = {max_fitness:.2f}, Avg LCI Balance = {avg_lci_balance:.2f}")
            
            # Select and reproduce
            self.selection_and_reproduction(fitness_results)
            
            # Save intermediate results every 10 generations
            if gen % 10 == 0 or gen == n_generations - 1:
                self.save_results(f"gen_{gen}")
        
        simulation_end_time = datetime.now()
        simulation_duration = simulation_end_time - simulation_start_time
        logger.info(f"Simulation completed in {simulation_duration}")
    
    def plot_results(self, filename: str = "lci_simulation_results.png"):
        """
        Generate plots of simulation results
        
        Args:
            filename: File name for the plot image
        """
        if not self.fitness_history:
            logger.warning("No fitness history available for plotting")
            return
            
        filepath = os.path.join(self.output_dir, filename)
        
        plt.figure(figsize=(15, 10))
        
        # Plot fitness history
        plt.subplot(2, 2, 1)
        plt.plot(self.fitness_history)
        plt.title('Average Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        
        # Plot LCI parameters over time
        plt.subplot(2, 2, 2)
        L_values, C_values, I_values = zip(*self.lci_history)
        plt.plot(L_values, label='L')
        plt.plot(C_values, label='C')
        plt.plot(I_values, label='I')
        plt.title('LCI Parameters Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot final population in LCI space
        plt.subplot(2, 2, 3)
        L = [agent.lci_params["L"] for agent in self.population]
        C = [agent.lci_params["C"] for agent in self.population]
        I = [agent.lci_params["I"] for agent in self.population]
        
        plt.scatter(L, C, c=I, cmap='viridis', alpha=0.7)
        plt.colorbar(label='I Value')
        plt.title('Final Population in L-C Space')
        plt.xlabel('L (Losslessness)')
        plt.ylabel('C (Compression)')
        
        # Plot final population LCI balance distribution
        plt.subplot(2, 2, 4)
        balance_values = [agent.get_lci_balance() for agent in self.population]
        plt.hist(balance_values, bins=20)
        plt.title('LCI Balance Distribution')
        plt.xlabel('LCI Balance')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(filepath)
        logger.info(f"Saved plot to {filepath}")
        
    def save_results(self, identifier: str = "final"):
        """
        Save simulation results to files
        
        Args:
            identifier: Identifier to append to file names
        """
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save best agent
        if self.best_agent_history:
            best_agent_ever = max(self.best_agent_history, key=lambda x: x["fitness"])
            best_agent_id = best_agent_ever["agent_id"]
            try:
                best_agent = next(a for a in self.population if a.agent_id == best_agent_id)
                model_path = os.path.join(self.output_dir, f"best_agent_{identifier}.pt")
                best_agent.save_model(model_path)
                logger.info(f"Saved best agent model to {model_path}")
            except StopIteration:
                logger.warning(f"Best agent {best_agent_id} not found in current population")
        
        # Save statistics
        stats = {
            "fitness_history": self.fitness_history,
            "lci_history": self.lci_history,
            "best_agent_history": self.best_agent_history,
            "env_config": self.env_config,
            "pop_size": self.pop_size,
            "mutation_rate": self.mutation_rate,
            "tournament_size": self.tournament_size,
            "elitism": self.elitism,
            "generation": self.generation,
        }
        
        stats_path = os.path.join(self.output_dir, f"stats_{identifier}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, default=lambda x: str(x) if isinstance(x, np.ndarray) else x)
        logger.info(f"Saved statistics to {stats_path}")
        
        # Generate and save plots
        self.plot_results(f"lci_results_{identifier}.png") 