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
import traceback

from lci_framework.agents.agent import TensorLCIAgent
from lci_framework.environments.environment import VectorizedMarkovEnvironment

# Configure logger
logger = logging.getLogger(__name__)

class TensorEvolution:
    """
    TensorEvolution implements a tensor-based evolutionary algorithm for optimizing LCI agents
    using GPU acceleration.
    
    Algorithmic Design:
    ------------------
    1. Selection Strategy: Uses tournament selection instead of alternatives like roulette wheel
       selection because tournament selection:
       - Has better selection pressure control via tournament size parameter
       - Is less sensitive to fitness scaling issues
       - Has O(n) complexity compared to O(n log n) for sorting-based methods
       
    2. Elitism: Preserves top performers to ensure monotonic improvement in best fitness.
       The implementation uses direct parameter copying rather than special agent marking
       to simplify state tracking.
       
    3. Mutation: Uses Gaussian noise with fixed standard deviation (0.1) applied to weights
       and biases. This approach was chosen over alternatives because:
       - It provides a balance between exploration and exploitation
       - It allows for continuous search space exploration
       - It's computationally efficient for tensor operations
       
    4. Fitness Calculation: Rewards are averaged by step count, which normalizes
       performance across different survival durations.
       
    5. Parallelization Strategy: Operations are vectorized where possible, falling back to
       sequential processing only when agent interactions cannot be batched.
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
                max_steps: int = 1000,
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
                learn_prob: float = 0.1,
                evolvable_energy: bool = True,
                output_dir: str = "results",
                device: Optional[torch.device] = None):
        """
        Initialize the tensor evolution framework.
        
        This refactored version is designed to work with TensorLCIAgent's
        population-based architecture directly, rather than treating it as
        a collection of individual agents.
        
        Args:
            pop_size: Number of agents in the population
            mutation_rate: Probability of mutation for each parameter
            tournament_size: Number of agents per tournament
            elitism: Number of top agents to preserve each generation
            n_states: Number of states in the environment
            n_actions: Number of actions in the environment
            n_generations: Number of generations to evolve
            steps_per_generation: Number of environment steps for each generation
            max_steps: Maximum number of steps to simulate (defaults to steps_per_generation)
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
            learn_prob: Learn probability for the agents
            evolvable_energy: Whether the agents' energy can evolve
            output_dir: Directory to save results
            device: Device for computation (auto-detected if None)
        """
        # Determine the device
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using specified device: mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using specified device: cuda")
        else:
            self.device = torch.device("cpu")
            logger.warning("No GPU available. Using CPU, which will be slow for tensor operations.")
        
        # Store parameters
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.n_generations = n_generations
        self.steps_per_generation = steps_per_generation
        self.max_steps = max_steps or steps_per_generation
        self.output_dir = output_dir
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Create vectorized environment
        self.env = VectorizedMarkovEnvironment(
            n_states=n_states,
            n_actions=n_actions,
            device=self.device
        )
        
        # Create population of agents (as a single TensorLCIAgent instance)
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
            learn_prob=learn_prob,
            evolvable_energy=evolvable_energy,
            device=self.device
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.generation = 0
        self.fitness_history = []  # Will store dicts with 'mean', 'std', 'max'
        self.lci_history = []      # Will store dicts with 'mean', 'max', 'alive_count'
        self.best_agent_history = []
        self.best_fitness = float('-inf')
        self.best_agent_generation = 0
        
        # For tracking energy warnings
        self._energy_warning_count = 0
        self._last_energy_warning_time = 0
        
        # For logging
        self.logger = logger
        
        logger.info(f"Tensor Evolution initialized with {pop_size} agents")
    
    def _monitor_agent_energy(self):
        """
        Monitor agent energy levels and provide diagnostics when issues occur.
        
        This method helps reduce log spam from energy warnings while still
        providing useful information about potential energy-related issues.
        """
        # Get current energy levels for all agents
        energy_levels = self.agents.get_energy()
        low_energy_count = torch.sum(energy_levels < self.agents.energy_cost_learn).item()
        
        # Only log warnings at a reasonable frequency
        current_time = time.time()
        if low_energy_count > 0 and (current_time - self._last_energy_warning_time > 10):  # At most one warning every 10 seconds
            self._energy_warning_count += 1
            self._last_energy_warning_time = current_time
            
            # Calculate statistics
            min_energy = energy_levels.min().item()
            mean_energy = energy_levels.mean().item()
            max_energy = energy_levels.max().item()
            
            # Convert energy costs to scalar values for safe string formatting
            learn_cost = self.agents.energy_cost_learn
            recovery_rate = self.agents.energy_recovery
            
            # Convert tensors to scalar values if necessary
            if isinstance(learn_cost, torch.Tensor):
                learn_cost = learn_cost.mean().item()
            if isinstance(recovery_rate, torch.Tensor):
                recovery_rate = recovery_rate.mean().item()
            
            # Log detailed warning first time, then less frequently
            if self._energy_warning_count == 1:
                logger.warning(
                    f"{low_energy_count} agents have insufficient energy for learning. "
                    f"Energy stats - Min: {min_energy:.4f}, Mean: {mean_energy:.4f}, Max: {max_energy:.4f}. "
                    f"Learning cost: {learn_cost:.4f}, Recovery: {recovery_rate:.4f}. "
                    f"Consider adjusting energy parameters if this persists."
                )
            elif self._energy_warning_count % 5 == 0:
                logger.warning(
                    f"Energy issues persist: {low_energy_count} agents below threshold. "
                    f"Min: {min_energy:.4f}, Mean: {mean_energy:.4f}"
                )
                
        return low_energy_count

    def evaluate_fitness(self):
        """
        Evaluate the fitness of the current population.
        
        Returns:
            Tuple of (fitness tensor, LCI values tensor)
        """
        # Initialize fitness tracking
        fitness_data = self._initialize_fitness_trackers()
        
        # Run simulation steps to evaluate fitness
        fitness_data = self._run_simulation_steps(fitness_data)
        
        # Calculate final fitness values
        return self._calculate_final_fitness(fitness_data)
    
    def _initialize_fitness_trackers(self):
        """
        Initialize the data structures for tracking fitness.
        
        Returns:
            Dictionary containing initialized fitness tracking data
        """
        # Create an immutable fitness data class
        from collections import namedtuple
        FitnessState = namedtuple('FitnessState', [
            'rewards', 'step_counts', 'alive_agents', 'observations', 
            'lci_values', 'n_alive'
        ])
        
        # Initialize tensors on the correct device
        rewards = self._zeros_like_population()
        step_counts = torch.zeros(self.pop_size, dtype=torch.int, device=self.device)
        alive_agents = torch.ones(self.pop_size, dtype=torch.bool, device=self.device)
        
        # Reset the environment for all agents
        observations = self.env.reset(pop_size=self.pop_size)
        
        # Get initial LCI values (zeros for now, will be updated during simulation)
        lci_values = self._zeros_like_population()
        
        # Track alive agents count
        n_alive = self.pop_size
        
        return FitnessState(
            rewards=rewards,
            step_counts=step_counts,
            alive_agents=alive_agents,
            observations=observations,
            lci_values=lci_values,
            n_alive=n_alive
        )

    def _run_simulation_steps(self, state):
        """
        Run the simulation steps to gather fitness data.
        
        Args:
            state: FitnessState containing fitness tracking data
            
        Returns:
            Updated FitnessState with fitness data after simulation
        """
        # Validate input
        required_attrs = ['rewards', 'step_counts', 'alive_agents', 'observations', 'lci_values', 'n_alive']
        if not all(hasattr(state, attr) for attr in required_attrs):
            missing_attrs = [attr for attr in required_attrs if not hasattr(state, attr)]
            logger.error(f"Missing required attributes in state: {missing_attrs}")
            return state
        
        # Create mutable copies of tensors we'll update
        rewards = state.rewards.clone()
        step_counts = state.step_counts.clone()
        alive_agents = state.alive_agents.clone()
        observations = state.observations.clone() if isinstance(state.observations, torch.Tensor) else state.observations
        lci_values = state.lci_values.clone()
        n_alive = state.n_alive
        
        # Initialize energy tracking tensor for this generation
        # Format: [step_idx, mean, min, max, std]
        # Pre-allocate for all steps plus initial state (steps_per_generation + 1)
        energy_tracking = torch.zeros((self.steps_per_generation + 1, 5), device=self.device)
        
        # Track initial energy
        energy_levels = self.agents.get_energy()
        energy_tracking[0, 0] = 0  # step 0
        energy_tracking[0, 1] = energy_levels.mean()
        energy_tracking[0, 2] = energy_levels.min()
        energy_tracking[0, 3] = energy_levels.max()
        energy_tracking[0, 4] = energy_levels.std() if energy_levels.numel() > 1 else 0.0
        
        # Simulate until all agents are dead or max steps reached
        step = 0
        while n_alive > 0 and step < self.steps_per_generation:
            try:
                # Monitor agent energy levels
                self._monitor_agent_energy()
                
                # Get actions from alive agents - TensorLCIAgent's predict method handles
                # filtering agents with energy internally
                actions = self._get_actions_from_population(observations, alive_agents)
                
                # Step the environment for all alive agents
                next_observations, reward_batch, done_batch, info = self.env.step_batch(actions)
                
                # Validate environment response
                if not (len(next_observations) == len(reward_batch) == len(done_batch) == self.pop_size):
                    logger.error(f"Environment step_batch returned inconsistent data sizes: " 
                                f"observations: {len(next_observations)}, rewards: {len(reward_batch)}, "
                                f"dones: {len(done_batch)}, expected: {self.pop_size}")
                    break
                
                # Update state information using population operations
                rewards, step_counts, alive_agents, observations, lci_values, n_alive = self._update_batch_state(
                    rewards, step_counts, alive_agents, observations, lci_values, n_alive,
                    actions, next_observations, reward_batch, done_batch
                )
                
                step += 1
                
                # Track energy after this step
                energy_levels = self.agents.get_energy()
                energy_tracking[step, 0] = step
                energy_tracking[step, 1] = energy_levels.mean()
                energy_tracking[step, 2] = energy_levels.min()
                energy_tracking[step, 3] = energy_levels.max()
                energy_tracking[step, 4] = energy_levels.std() if energy_levels.numel() > 1 else 0.0
                
                # Periodic logging
                if step % 50 == 0:
                    logger.debug(f"Step {step}: {n_alive} agents alive, Mean energy: {energy_levels.mean().item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error during simulation step {step}: {str(e)}")
                break
        
        # Store this generation's energy tracking (only up to the steps we actually ran)
        valid_steps = step + 1  # +1 to include step 0
        
        # Initialize step_energy_tracking if not exists
        if not hasattr(self, 'step_energy_tracking'):
            self.step_energy_tracking = []
            
        # Convert tensor to list of dictionaries for JSON serialization (only at the end)
        generation_tracking = []
        for i in range(valid_steps):
            generation_tracking.append({
                'step': int(energy_tracking[i, 0].item()),
                'mean': energy_tracking[i, 1].item(),
                'min': energy_tracking[i, 2].item(),
                'max': energy_tracking[i, 3].item(),
                'std': energy_tracking[i, 4].item()
            })
            
        self.step_energy_tracking.append(generation_tracking)
        
        # Create a new state with updated values
        from collections import namedtuple
        FitnessState = namedtuple('FitnessState', [
            'rewards', 'step_counts', 'alive_agents', 'observations', 
            'lci_values', 'n_alive'
        ])
        
        return FitnessState(
            rewards=rewards,
            step_counts=step_counts,
            alive_agents=alive_agents,
            observations=observations,
            lci_values=lci_values,
            n_alive=n_alive
        )

    def _get_actions_from_population(self, observations, alive_agents):
        """
        Get actions from the agent population using batch operations.
        
        Args:
            observations: Current observations for all agents
            alive_agents: Boolean tensor indicating which agents are alive
            
        Returns:
            Tensor of actions for all agents
        """
        try:
            # Ensure observations are in the right shape for batch processing
            if isinstance(observations, torch.Tensor) and observations.dim() == 2:
                # Already in correct shape [pop_size, observation_size]
                pass
            elif isinstance(observations, list) and len(observations) == self.pop_size:
                # Convert list to tensor
                observations = torch.stack([obs.to(self.device) if isinstance(obs, torch.Tensor) 
                                           else torch.tensor(obs, device=self.device) 
                                           for obs in observations])
            else:
                logger.error(f"Observations in unexpected format: {type(observations)}")
                return torch.zeros(self.pop_size, dtype=torch.long, device=self.device)
                
            # Get predictions from the model using all agents (not just alive ones)
            # This avoids tensor shape mismatch issues when some agents die
            with torch.no_grad():
                # Pass alive_agents mask to the predict method
                predictions = self.agents.predict(observations, alive_mask=alive_agents)
                
                # Convert to action indices
                actions = (predictions.squeeze(-1) >= 0.5).long()
                
                # For agents that aren't alive, use random actions (they'll be ignored in updates)
                mask = ~alive_agents
                if mask.any():
                    random_actions = torch.randint(0, self.n_actions, (mask.sum(),), device=self.device)
                    actions[mask] = random_actions
                
                return actions
        except Exception as e:
            logger.error(f"Error getting agent actions: {str(e)}")
            # Return default actions for the full population
            return torch.zeros(self.pop_size, dtype=torch.long, device=self.device)

    def _update_batch_state(self, rewards, step_counts, alive_agents, observations, 
                           lci_values, n_alive, actions, next_observations, 
                           reward_batch, done_batch):
        """
        Update simulation state based on environment step results,
        handling the entire population as a batch.
        
        Returns:
            Tuple of updated state tensors
        """
        # Verify tensor dimensions
        if rewards.shape[0] != self.pop_size or step_counts.shape[0] != self.pop_size:
            logger.warning(f"Tensor dimension mismatch in _update_batch_state. "
                          f"Expected pop_size={self.pop_size}, got rewards={rewards.shape}, "
                          f"step_counts={step_counts.shape}")
            # Ensure dimensions match by recreating tensors if needed
            rewards = torch.zeros(self.pop_size, device=self.device)
            step_counts = torch.zeros(self.pop_size, dtype=torch.int, device=self.device)
            alive_agents = torch.ones(self.pop_size, dtype=torch.bool, device=self.device)
            lci_values = torch.zeros(self.pop_size, device=self.device)
        
        # Use in-place operations where possible to reduce memory usage
        
        # Update rewards for alive agents (in-place)
        rewards = torch.where(alive_agents, rewards + reward_batch, rewards)
        
        # Update step counts for alive agents (in-place)
        step_counts = torch.where(alive_agents, step_counts + 1, step_counts)
        
        # Get energy levels to check which agents are still alive
        energy = self.agents.get_energy()
        
        # Calculate performance factors (reward relative to maximum possible)
        performance_factor = torch.clamp(reward_batch / 1.0, min=0.0, max=1.0)
        
        # Learn from experience using batched operations
        if alive_agents.any():
            # Determine which agents should learn (vectorized operation)
            learn_prob = getattr(self.agents, 'learn_probability', 0.1)
            learn_rand = torch.rand(self.pop_size, device=self.device)
            
            if isinstance(learn_prob, torch.Tensor):
                # Use each agent's individual learn probability
                learn_mask = alive_agents & (learn_rand < learn_prob)
            else:
                # Use fixed learn probability
                learn_mask = alive_agents & (learn_rand < learn_prob)
            
            # Only attempt learning if at least one agent should learn
            if learn_mask.any():
                # Let the agent's learn method handle batching and energy checks
                try:
                    # Pass alive_agents mask to learn method
                    self.agents.learn(observations, reward_batch, alive_mask=alive_agents)
                except Exception as e:
                    logger.warning(f"Learning failed: {e}. Continuing with simulation.")
        
        # Update agent energy recovery with performance factor (batched)
        self.agents.update_energy(performance_factor)
        
        # Get updated energy after recovery
        energy = self.agents.get_energy()
        
        # Update alive status based on done flag and energy (vectorized)
        new_alive_agents = alive_agents & ~done_batch & (energy > 0)
        n_alive = new_alive_agents.sum().item()
        
        # Update observations for agents that are still alive (fully vectorized)
        if isinstance(observations, torch.Tensor) and isinstance(next_observations, torch.Tensor):
            # Verify observation tensor dimensions
            if observations.shape[0] != self.pop_size or next_observations.shape[0] != self.pop_size:
                logger.warning(f"Observation dimension mismatch. Expected pop_size={self.pop_size}, "
                               f"got observations={observations.shape}, next_observations={next_observations.shape}")
                # Reset observations to default
                observations = self._get_default_observations()
            else:
                # Create a mask tensor that's the same shape as observations
                mask = new_alive_agents.unsqueeze(1).expand_as(observations)
                observations = torch.where(mask, next_observations, observations)
        else:
            # Handle non-tensor observations if absolutely necessary
            for i in range(min(len(observations), len(next_observations), self.pop_size)):
                if i < len(new_alive_agents) and new_alive_agents[i]:
                    observations[i] = next_observations[i]
        
        # Update LCI values (vectorized)
        lci_values = rewards / torch.clamp(step_counts.float(), min=1)
        
        # Update alive agents
        alive_agents = new_alive_agents
        
        return rewards, step_counts, alive_agents, observations, lci_values, n_alive
        
    def _get_default_observations(self):
        """
        Get default observations for the entire population.
        Used when tensor dimensions don't match.
        
        Returns:
            Tensor of default observations
        """
        # We'll use the environment's reset method to get default observations
        return self.env.reset(self.pop_size)

    def _calculate_final_fitness(self, fitness_data):
        """
        Calculate final fitness values based on rewards and other factors.
        
        Args:
            fitness_data: FitnessState containing fitness tracking data
            
        Returns:
            Tuple of (fitness tensor, LCI values tensor)
        """
        try:
            # Extract values needed for calculation
            rewards = fitness_data.rewards
            step_counts = fitness_data.step_counts
            lci_values = fitness_data.lci_values
            
            # Avoid division by zero
            step_counts_safe = torch.clamp(step_counts, min=1)
            
            # Calculate fitness as average reward per step (pure calculation)
            fitness = self._calculate_fitness_from_rewards(rewards, step_counts_safe)
            
            # Calculate statistics (pure calculation)
            stats = self._calculate_statistics(fitness, lci_values, step_counts)
            
            # Log statistics
            self._log_generation_stats(stats)
            
            # Store statistics for later analysis (side effect, but isolated)
            self._record_history(stats)
            
            return fitness, lci_values
        except Exception as e:
            logger.error(f"Error calculating fitness: {str(e)}")
            # Return default values in case of error
            return torch.zeros(self.pop_size, device=self.device), torch.zeros(self.pop_size, device=self.device)
    
    def _calculate_fitness_from_rewards(self, rewards, step_counts_safe):
        """
        Calculate fitness from rewards and step counts.
        
        Args:
            rewards: Tensor of rewards for each agent
            step_counts_safe: Tensor of step counts for each agent (clamped to avoid division by zero)
            
        Returns:
            Tensor of fitness values for each agent
        """
        return rewards / step_counts_safe.float()
    
    def _calculate_statistics(self, fitness, lci_values, step_counts):
        """
        Calculate statistics for the current generation.
        
        Args:
            fitness: Tensor of fitness values for each agent
            lci_values: Tensor of LCI values for each agent
            step_counts: Tensor of step counts for each agent
            
        Returns:
            Named tuple with statistics
        """
        from collections import namedtuple
        Stats = namedtuple('Stats', [
            'generation', 'mean_fitness', 'std_fitness', 'max_fitness',
            'mean_lci', 'max_lci', 'alive_count', 'best_idx'
        ])
        
        # Find the agent with highest fitness
        max_fitness_value, best_idx = torch.max(fitness, dim=0)
        
        # Get alive count directly from agent energy levels
        # Only count agents with enough energy to learn as "alive"
        energy_levels = self.agents.get_energy()
        alive_count = (energy_levels >= self.agents.energy_cost_learn).sum().item()
        
        return Stats(
            generation=self.generation,
            mean_fitness=fitness.mean().item(),
            std_fitness=fitness.std().item() if fitness.numel() > 1 else 0.0,
            max_fitness=max_fitness_value.item(),
            mean_lci=lci_values.mean().item(),
            max_lci=lci_values.max().item(),
            alive_count=alive_count,
            best_idx=best_idx.item()
        )
    
    def _log_generation_stats(self, stats):
        """
        Log statistics for the current generation.
        
        Args:
            stats: Named tuple with statistics
        """
        logger.info(
            f"Generation {stats.generation}: "
            f"Mean fitness: {stats.mean_fitness:.4f}, "
            f"Max fitness: {stats.max_fitness:.4f}, "
            f"Alive: {stats.alive_count}/{self.pop_size}"
        )
    
    def _record_history(self, stats):
        """
        Record statistics for later analysis.
        
        Args:
            stats: Named tuple with statistics
        """
        # Record fitness statistics
        self.fitness_history.append({
            'generation': stats.generation,
            'mean': stats.mean_fitness,
            'std': stats.std_fitness,
            'max': stats.max_fitness
        })
        
        # Record LCI statistics
        self.lci_history.append({
            'generation': stats.generation,
            'mean': stats.mean_lci,
            'max': stats.max_lci,
            'alive_count': stats.alive_count
        })
        
        # Record energy statistics
        energy_levels = self.agents.get_energy()
        if not hasattr(self, 'energy_history'):
            self.energy_history = []
            
        self.energy_history.append({
            'generation': stats.generation,
            'mean': energy_levels.mean().item(),
            'min': energy_levels.min().item(),
            'max': energy_levels.max().item(),
            'std': energy_levels.std().item() if energy_levels.numel() > 1 else 0.0
        })
        
        # Check if this is the best fitness seen so far
        if stats.max_fitness > self.best_fitness:
            self.best_fitness = stats.max_fitness
            self.best_agent_generation = stats.generation
            
            # Record best agent
            self.best_agent_history.append({
                'generation': stats.generation,
                'fitness': stats.max_fitness,
                'lci': self.agents.get_lci().max().item()  # Use get_lci() method instead of lci attribute
            })

    def _run_tournament_selection(self, fitness):
        """
        Run tournament selection to select parents for reproduction.
        
        Args:
            fitness: Tensor of fitness values for the population
            
        Returns:
            Indices of the winners from each tournament
        """
        # This method is deprecated, use the vectorized version instead
        return self._vectorized_tournament_selection(fitness)

    def _vectorized_tournament_selection(self, fitness):
        """
        Vectorized tournament selection that uses tensor operations to run all tournaments in parallel.
        
        Args:
            fitness: Tensor of fitness values for the population [pop_size]
            
        Returns:
            Indices of the winners from each tournament [pop_size]
        """
        # Pre-allocate memory for all tournaments at once [pop_size, tournament_size]
        tournament_indices = torch.randint(
            0, self.pop_size, 
            (self.pop_size, self.tournament_size), 
            device=self.device
        )
        
        # Use advanced indexing to get fitness of all participants in one operation
        tournament_fitness = fitness[tournament_indices]
        
        # Find winners using argmax (highest fitness) in each tournament
        winner_relative_indices = torch.argmax(tournament_fitness, dim=1)
        
        # Get actual population indices of winners using gather
        return tournament_indices.gather(1, winner_relative_indices.unsqueeze(1)).squeeze(1)

    def selection_and_reproduction(self, fitness: torch.Tensor) -> None:
        """
        Performs selection, reproduction and mutation to create the next generation
        using fully batched tensor operations.
        
        Args:
            fitness: A tensor containing the fitness values for each agent in the population.
        """
        try:
            # Calculate how many elite agents to keep
            num_elite = max(1, int(self.elitism * self.pop_size) if isinstance(self.elitism, float) else self.elitism)
            
            # Get indices of agents with top fitness
            _, elite_indices = torch.topk(fitness, num_elite)
            
            # Select parents through tournament selection (fully vectorized)
            parent_indices = self._vectorized_tournament_selection(fitness)
            
            # Create new population by copying parameters from parents in a single batch operation
            for layer_idx in range(self.agents.model.n_layers):
                # Get all weights and biases for this layer
                weights, biases = self.agents.get_layer_parameters(layer_idx)
                
                # Create new weights and biases by selecting from parents (all in one batch)
                new_weights = weights[parent_indices]
                new_biases = biases[parent_indices]
                
                # Apply mutations to all agents at once
                with torch.no_grad():
                    # Mutation mask (True where mutation should occur)
                    weight_mutation_mask = torch.rand_like(new_weights) < self.mutation_rate
                    bias_mutation_mask = torch.rand_like(new_biases) < self.mutation_rate
                    
                    # Create Gaussian noise for mutations, matching the shape of weights and biases
                    weight_mutations = torch.randn_like(new_weights) * 0.1
                    bias_mutations = torch.randn_like(new_biases) * 0.1
                    
                    # Apply mutations using tensor operations
                    new_weights = new_weights + (weight_mutation_mask * weight_mutations)
                    new_biases = new_biases + (bias_mutation_mask * bias_mutations)
                    
                    # Preserve elite agents' parameters
                    if num_elite > 0:
                        # Use advanced indexing to copy parameters from elite agents back to elite positions
                        # This replaces mutations for the elite agents
                        new_weights[:num_elite] = weights[elite_indices]
                        new_biases[:num_elite] = biases[elite_indices]
                    
                    # Update the layer with new parameters (entire population at once)
                    self.agents.set_layer_parameters(layer_idx, new_weights, new_biases)
            
            # Handle output layer separately but still in a fully batched manner
            out_weights, out_biases = self.agents.get_output_parameters()
            new_out_weights = out_weights[parent_indices]
            new_out_biases = out_biases[parent_indices]
            
            with torch.no_grad():
                # Mutation mask for output layer
                out_weight_mutation_mask = torch.rand_like(new_out_weights) < self.mutation_rate
                out_bias_mutation_mask = torch.rand_like(new_out_biases) < self.mutation_rate
                
                # Create Gaussian noise for mutations
                out_weight_mutations = torch.randn_like(new_out_weights) * 0.1
                out_bias_mutations = torch.randn_like(new_out_biases) * 0.1
                
                # Apply mutations using tensor operations
                new_out_weights = new_out_weights + (out_weight_mutation_mask * out_weight_mutations)
                new_out_biases = new_out_biases + (out_bias_mutation_mask * out_bias_mutations)
                
                # Preserve elite agents' parameters
                if num_elite > 0:
                    new_out_weights[:num_elite] = out_weights[elite_indices]
                    new_out_biases[:num_elite] = out_biases[elite_indices]
                
                # Update output layer parameters (entire population at once)
                self.agents.set_output_parameters(new_out_weights, new_out_biases)
            
            # Handle evolvable energy parameters if they exist
            if hasattr(self.agents, 'evolvable_energy') and self.agents.evolvable_energy:
                # Copy energy genes from parents
                new_energy_efficiency = self.agents.energy_efficiency[parent_indices]
                new_learn_probability = self.agents.learn_probability[parent_indices]
                new_recovery_rate = self.agents.recovery_rate[parent_indices]
                
                # Apply mutations
                efficiency_mutation_mask = torch.rand_like(new_energy_efficiency) < self.mutation_rate
                learn_prob_mutation_mask = torch.rand_like(new_learn_probability) < self.mutation_rate
                recovery_mutation_mask = torch.rand_like(new_recovery_rate) < self.mutation_rate
                
                # Create mutations
                efficiency_mutations = 0.1 * torch.randn_like(new_energy_efficiency)
                learn_prob_mutations = 0.02 * torch.randn_like(new_learn_probability)
                recovery_mutations = 0.005 * torch.randn_like(new_recovery_rate)
                
                # Apply mutations
                new_energy_efficiency = new_energy_efficiency + (efficiency_mutation_mask * efficiency_mutations)
                new_learn_probability = new_learn_probability + (learn_prob_mutation_mask * learn_prob_mutations)
                new_recovery_rate = new_recovery_rate + (recovery_mutation_mask * recovery_mutations)
                
                # Preserve elite genes
                if num_elite > 0:
                    new_energy_efficiency[:num_elite] = self.agents.energy_efficiency[elite_indices]
                    new_learn_probability[:num_elite] = self.agents.learn_probability[elite_indices]
                    new_recovery_rate[:num_elite] = self.agents.recovery_rate[elite_indices]
                
                # Ensure valid values
                new_energy_efficiency = torch.clamp(new_energy_efficiency, min=0.5, max=1.5)
                new_learn_probability = torch.clamp(new_learn_probability, min=0.01, max=0.3)
                new_recovery_rate = torch.clamp(new_recovery_rate, min=0.005, max=0.1)
                
                # Update agent's energy genes
                self.agents.energy_efficiency = new_energy_efficiency
                self.agents.learn_probability = new_learn_probability
                self.agents.recovery_rate = new_recovery_rate
                
                # Update derived values
                self.agents.energy_cost_predict = torch.full((self.pop_size,), self.agents.base_energy_cost_predict, 
                                                          device=self.device) / self.agents.energy_efficiency
                self.agents.energy_cost_learn = torch.full((self.pop_size,), self.agents.base_energy_cost_learn, 
                                                        device=self.device) / self.agents.energy_efficiency
                self.agents.energy_recovery = self.agents.recovery_rate
            
            # Reset agent state variables for next generation
            self.agents.reset_state()
        
        except Exception as e:
            logger.error(f"Error in selection and reproduction: {str(e)}")
            traceback.print_exc()

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
        
        # Track the best agent (moved to _calculate_final_fitness)
        best_idx = torch.argmax(fitness).item()
        best_fitness_gen = fitness[best_idx].item()
        
        if best_fitness_gen > self.best_fitness:
            self.best_fitness = best_fitness_gen
            self.best_agent_generation = self.generation
            
            # Save the best agent
            os.makedirs(f"{self.output_dir}/best_agents", exist_ok=True)
            self.agents.save_agent(f"{self.output_dir}/best_agents/best_agent_gen_{self.generation}")
        
        # Statistics tracking is now done in _calculate_final_fitness
        
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
        self.energy_history = []
        self.best_agent_history = []
        self.best_fitness = float('-inf')
        self.best_agent_generation = 0
        
        # Clear step energy tracking
        self.step_energy_tracking = []
        
        # Run for n_generations
        for gen in range(self.n_generations):
            # Run a single generation
            self.run_generation(self.steps_per_generation)
            
            # Save current results periodically
            if gen % 10 == 0 or gen == self.n_generations - 1:
                self.save_results()
                self.plot_results()
                self.plot_step_energy()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best agent found in generation {self.best_agent_generation} with fitness {self.best_fitness:.4f}")
        
        # Save final results
        self.save_results()
        self.plot_results()
        self.plot_step_energy()
        
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
        # Create figure with 3 subplots (adding energy plot)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
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
        
        # Plot energy history if available
        if hasattr(self, 'energy_history') and self.energy_history:
            mean_energy = [gen['mean'] for gen in self.energy_history]
            min_energy = [gen['min'] for gen in self.energy_history]
            max_energy = [gen['max'] for gen in self.energy_history]
            
            # Get scalar values for thresholds
            learn_threshold = self._get_scalar_energy_value(self.agents.energy_cost_learn)
            predict_threshold = self._get_scalar_energy_value(self.agents.energy_cost_predict)
            
            ax3.plot(generations, mean_energy, label='Mean Energy', color='blue')
            ax3.plot(generations, min_energy, label='Min Energy', color='red')
            ax3.plot(generations, max_energy, label='Max Energy', color='green')
            ax3.axhline(y=learn_threshold, color='orange', linestyle='--', 
                      label=f'Learning Threshold ({learn_threshold:.3f})')
            ax3.axhline(y=predict_threshold, color='purple', linestyle=':', 
                      label=f'Prediction Threshold ({predict_threshold:.3f})')
            ax3.set_title('Agent Energy Levels')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Energy')
            ax3.legend()
            ax3.grid(True)
        
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
        
        # Add energy history if available
        if hasattr(self, 'energy_history') and self.energy_history:
            results['energy_history'] = self.energy_history
            
        # Add step-by-step energy tracking if available
        if hasattr(self, 'step_energy_tracking') and self.step_energy_tracking:
            results['step_energy_tracking'] = self.step_energy_tracking
        
        with open(f"{self.output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

    def _zeros_like_population(self):
        """Create a zero tensor with population size on the correct device."""
        return torch.zeros(self.pop_size, device=self.device)
    
    def _ones_like_population(self):
        """Create a ones tensor with population size on the correct device."""
        return torch.ones(self.pop_size, device=self.device)

    def _get_scalar_energy_value(self, energy_param):
        """Get a scalar value from an energy parameter that could be either a tensor or a scalar."""
        if isinstance(energy_param, torch.Tensor):
            # If it's a tensor with multiple values (one per agent), use the mean
            if energy_param.numel() > 1:
                return energy_param.mean().item()
            # If it's a single value tensor, just get the item
            return energy_param.item()
        # If it's already a scalar, return it directly
        return energy_param
        
    def plot_step_energy(self) -> None:
        """
        Plot the detailed energy tracking over steps within each generation.
        """
        if not hasattr(self, 'step_energy_tracking') or not self.step_energy_tracking:
            logger.warning("No step energy tracking data available for plotting")
            return
            
        # Create figure with one subplot per generation
        n_generations = len(self.step_energy_tracking)
        fig, axes = plt.subplots(n_generations, 1, figsize=(10, 5 * n_generations))
        
        # Handle single generation case
        if n_generations == 1:
            axes = [axes]
        
        # Get scalar values for thresholds
        learn_threshold = self._get_scalar_energy_value(self.agents.energy_cost_learn)
        predict_threshold = self._get_scalar_energy_value(self.agents.energy_cost_predict)
            
        # Plot each generation's energy tracking
        for i, gen_data in enumerate(self.step_energy_tracking):
            ax = axes[i]
            
            # Extract data
            steps = [entry['step'] for entry in gen_data]
            mean_energy = [entry['mean'] for entry in gen_data]
            min_energy = [entry['min'] for entry in gen_data]
            max_energy = [entry['max'] for entry in gen_data]
            
            # Plot energy levels
            ax.plot(steps, mean_energy, 'b-', label='Mean Energy')
            ax.plot(steps, min_energy, 'r-', label='Min Energy')
            ax.plot(steps, max_energy, 'g-', label='Max Energy')
            
            # Add horizontal lines for energy thresholds
            ax.axhline(y=learn_threshold, color='orange', linestyle='--', 
                      label=f'Learning Threshold ({learn_threshold:.3f})')
            ax.axhline(y=predict_threshold, color='purple', linestyle=':', 
                      label=f'Prediction Threshold ({predict_threshold:.3f})')
            
            # Add labels and title
            ax.set_title(f'Energy Levels During Generation {i}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Energy')
            ax.legend()
            ax.grid(True)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/step_energy_tracking.png")
        plt.close() 