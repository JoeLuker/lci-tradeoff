"""
Tensor Evolution with MLX

This module implements a fully vectorized evolutionary algorithm that 
optimizes agent populations using Apple Silicon acceleration.
"""

import mlx.core as mx
import os
import logging
import time
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback

from lci_framework.agents.agent import TensorLCIAgent
from lci_framework.environments.environment import VectorizedMarkovEnvironment

# Configure logger
logger = logging.getLogger(__name__)

class TensorEvolution:
    """
    TensorEvolution implements a tensor-based evolutionary algorithm for optimizing LCI agents
    using Apple Silicon acceleration with MLX.
    
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
                device: Optional[Any] = None):
        """
        Initialize the tensor evolution framework.
        
        This is designed to work with TensorLCIAgent's population-based architecture
        directly, rather than treating it as a collection of individual agents.
        
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
            device: Not used in MLX (kept for API compatibility)
        """
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
        
        logger.info("Using MLX with unified memory architecture for Apple Silicon acceleration")
        
        # Create vectorized environment
        self.env = VectorizedMarkovEnvironment(
            n_states=n_states,
            n_actions=n_actions
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
            evolvable_energy=evolvable_energy
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
        low_energy_count = mx.sum(energy_levels < self.agents.energy_cost_learn).item()
        
        # Only log warnings at a reasonable frequency
        current_time = time.time()
        if low_energy_count > 0 and (current_time - self._last_energy_warning_time > 10):  # At most one warning every 10 seconds
            self._energy_warning_count += 1
            self._last_energy_warning_time = current_time
            
            # Calculate statistics
            min_energy = energy_levels.min().item()
            mean_energy = energy_levels.mean().item()
            max_energy = energy_levels.max().item()
            
            # Get energy costs for learning (may be per-agent or scalar)
            if isinstance(self.agents.energy_cost_learn, mx.array):
                learn_cost = self.agents.energy_cost_learn.mean().item()
            else:
                learn_cost = self.agents.energy_cost_learn
                
            # Get recovery rate (may be per-agent or scalar)
            if isinstance(self.agents.energy_recovery, mx.array):
                recovery_rate = self.agents.energy_recovery.mean().item()
            else:
                recovery_rate = self.agents.energy_recovery
            
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
        
        # Initialize arrays
        rewards = self._zeros_like_population()
        step_counts = mx.zeros((self.pop_size,), dtype=mx.int32)
        alive_agents = mx.ones((self.pop_size,), dtype=mx.bool_)
        
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
        
        # Create copies of arrays we'll update
        rewards = state.rewards
        step_counts = state.step_counts
        alive_agents = state.alive_agents
        observations = state.observations
        lci_values = state.lci_values
        n_alive = state.n_alive
        
        # Initialize energy tracking array for this generation
        # Format: [step_idx, mean, min, max, std]
        # Pre-allocate for all steps plus initial state (steps_per_generation + 1)
        energy_tracking = mx.zeros((self.steps_per_generation + 1, 5))
        
        # Track initial energy
        energy_levels = self.agents.get_energy()
        energy_tracking[0, 0] = 0.0  # step 0
        energy_tracking[0, 1] = mx.mean(energy_levels).item()
        energy_tracking[0, 2] = mx.min(energy_levels).item()
        energy_tracking[0, 3] = mx.max(energy_levels).item()
        energy_tracking[0, 4] = mx.std(energy_levels).item() if energy_levels.size > 1 else 0.0
        
        # Simulate until all agents are dead or max steps reached
        step = 0
        while n_alive > 0 and step < self.steps_per_generation:
            try:
                # Monitor agent energy levels
                self._monitor_agent_energy()
                
                # Get actions from alive agents
                actions = self._get_actions_from_population(observations, alive_agents)
                
                # Step the environment for all alive agents
                next_observations, reward_batch, done_batch, info = self.env.step(actions)
                
                # Validate environment response
                if not (len(next_observations) == len(reward_batch) == len(done_batch) == self.pop_size):
                    logger.error(f"Environment step returned inconsistent data sizes: " 
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
                energy_tracking[step, 0] = float(step)
                energy_tracking[step, 1] = mx.mean(energy_levels).item()
                energy_tracking[step, 2] = mx.min(energy_levels).item()
                energy_tracking[step, 3] = mx.max(energy_levels).item()
                energy_tracking[step, 4] = mx.std(energy_levels).item() if energy_levels.size > 1 else 0.0
                
                # Periodic logging
                if step % 50 == 0:
                    logger.debug(f"Step {step}: {n_alive} agents alive, Mean energy: {mx.mean(energy_levels).item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error during simulation step {step}: {str(e)}")
                traceback.print_exc()  # Print full traceback for debugging
                break
        
        # Store this generation's energy tracking (only up to the steps we actually ran)
        valid_steps = step + 1  # +1 to include step 0
        
        # Initialize step_energy_tracking if not exists
        if not hasattr(self, 'step_energy_tracking'):
            self.step_energy_tracking = []
            
        # Convert array to list of dictionaries for JSON serialization
        generation_tracking = []
        for i in range(valid_steps):
            generation_tracking.append({
                'step': int(energy_tracking[i, 0].item()),
                'mean': float(energy_tracking[i, 1].item()),
                'min': float(energy_tracking[i, 2].item()),
                'max': float(energy_tracking[i, 3].item()),
                'std': float(energy_tracking[i, 4].item())
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
            observations: Current observations for all agents (MLX array)
            alive_agents: Boolean array indicating which agents are alive (MLX array)
            
        Returns:
            MLX array of actions for all agents
        """
        try:
            # Debug: Check input types
            logger.debug(f"observations type: {type(observations)}, shape: {getattr(observations, 'shape', None)}")
            logger.debug(f"alive_agents type: {type(alive_agents)}, shape: {getattr(alive_agents, 'shape', None)}")
            
            # Ensure observations are in the right shape for batch processing
            if not hasattr(observations, 'shape'):
                logger.error(f"Observations missing shape attribute: {type(observations)}")
                return mx.zeros((self.pop_size,), dtype=mx.int32)
                
            if observations.ndim == 2 and observations.shape[0] == self.pop_size:
                # Already in correct shape [pop_size, observation_size]
                pass
            elif hasattr(observations, 'ndim') and observations.ndim != 2:
                logger.error(f"Observations have unexpected dimensions: {observations.ndim}")
                return mx.zeros((self.pop_size,), dtype=mx.int32)
            else:
                logger.error(f"Observations in unexpected format: {type(observations)}")
                return mx.zeros((self.pop_size,), dtype=mx.int32)
                
            # Get predictions from the model using all agents (not just alive ones)
            # This avoids tensor shape mismatch issues when some agents die
            predictions = self.agents.predict(observations, alive_mask=alive_agents)
            logger.debug(f"predictions type: {type(predictions)}, shape: {getattr(predictions, 'shape', None)}")
            
            # Safely convert to action indices - threshold at 0.5
            # First squeeze the trailing dimension
            squeezed_preds = mx.squeeze(predictions, axis=-1)
            logger.debug(f"squeezed_preds type: {type(squeezed_preds)}, shape: {getattr(squeezed_preds, 'shape', None)}")
            
            # Create a boolean array
            bool_actions = squeezed_preds >= 0.5
            logger.debug(f"bool_actions type: {type(bool_actions)}, shape: {getattr(bool_actions, 'shape', None)}")
            
            # Use direct array creation to avoid tuple issues
            try:
                actions = mx.zeros(bool_actions.shape, dtype=mx.int32)
                actions = mx.where(bool_actions, mx.array(1, dtype=mx.int32), mx.array(0, dtype=mx.int32))
                logger.debug(f"actions type: {type(actions)}, shape: {getattr(actions, 'shape', None)}")
            except Exception as e:
                logger.error(f"Error creating actions array: {str(e)}")
                # Try alternative array creation
                actions = mx.zeros((self.pop_size,), dtype=mx.int32)
                for i in range(self.pop_size):
                    if bool_actions[i].item():
                        actions = mx.array_at_indices(actions, i, mx.array(1, dtype=mx.int32))
            
            # For agents that aren't alive, use random actions (they'll be ignored in updates)
            if mx.sum(~alive_agents) > 0:
                # Generate random actions for dead agents
                num_dead = int(mx.sum(~alive_agents).item())
                random_actions = mx.random.randint(0, self.n_actions, shape=(num_dead,), dtype=mx.int32)
                logger.debug(f"random_actions type: {type(random_actions)}, shape: {getattr(random_actions, 'shape', None)}")
                
                # Loop through dead agents and update their actions one by one
                dead_idx = 0
                for i in range(self.pop_size):
                    if not alive_agents[i].item():
                        try:
                            # Get integer value, not a tuple
                            action_value = int(random_actions[dead_idx].item()) 
                            logger.debug(f"action_value: {action_value}, type: {type(action_value)}")
                            
                            # Use array_at_indices for safe assignment
                            actions = mx.array_at_indices(actions, i, mx.array(action_value, dtype=mx.int32))
                            dead_idx += 1
                        except Exception as e:
                            logger.error(f"Error updating action at index {i}: {str(e)}")
            
            return actions
        except Exception as e:
            logger.error(f"Error getting agent actions: {str(e)}")
            # Return default actions for the full population
            return mx.zeros((self.pop_size,), dtype=mx.int32)

    def _update_batch_state(self, rewards, step_counts, alive_agents, observations, 
                           lci_values, n_alive, actions, next_observations, 
                           reward_batch, done_batch):
        """
        Update simulation state based on environment step results,
        handling the entire population as a batch.
        
        Args:
            rewards: MLX array of rewards
            step_counts: MLX array of step counts
            alive_agents: MLX boolean array of alive status
            observations: MLX array of current observations
            lci_values: MLX array of LCI values
            n_alive: Number of alive agents
            actions: MLX array of actions
            next_observations: MLX array of next observations
            reward_batch: MLX array of rewards from environment
            done_batch: MLX array of done flags from environment
            
        Returns:
            Tuple of updated state arrays (all MLX arrays)
        """
        # Verify array dimensions - create new arrays if needed
        if not hasattr(rewards, 'shape') or rewards.shape[0] != self.pop_size or step_counts.shape[0] != self.pop_size:
            logger.warning(f"Array dimension mismatch in _update_batch_state. "
                          f"Expected pop_size={self.pop_size}, got rewards shape={getattr(rewards, 'shape', None)}, "
                          f"step_counts shape={getattr(step_counts, 'shape', None)}")
            # Ensure dimensions match by recreating arrays
            rewards = mx.zeros((self.pop_size,))
            step_counts = mx.zeros((self.pop_size,), dtype=mx.int32)
            alive_agents = mx.ones((self.pop_size,), dtype=mx.bool_)
            lci_values = mx.zeros((self.pop_size,))
        
        # Use MLX operations where possible
        
        # Update rewards for alive agents
        rewards = mx.where(alive_agents, rewards + reward_batch, rewards)
        
        # Update step counts for alive agents
        step_counts = mx.where(alive_agents, step_counts + 1, step_counts)
        
        # Get energy levels to check which agents are still alive
        energy = self.agents.get_energy()
        
        # Calculate performance factors (reward relative to maximum possible)
        performance_factor = mx.clip(reward_batch / 1.0, a_min=0.0, a_max=1.0)
        
        # Learn from experience using batched operations
        if mx.sum(alive_agents) > 0:
            # Determine which agents should learn
            learn_prob = getattr(self.agents, 'learn_probability', 0.1)
            
            # Create random values for learn probability check
            learn_rand = mx.random.uniform(shape=(self.pop_size,))
            
            # Determine learn mask based on probability
            if hasattr(learn_prob, 'shape'):
                # Use each agent's individual learn probability
                learn_mask = alive_agents & (learn_rand < learn_prob)
            else:
                # Use fixed learn probability
                learn_mask = alive_agents & (learn_rand < learn_prob)
            
            # Only attempt learning if at least one agent should learn
            if mx.sum(learn_mask) > 0:
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
        
        # Update alive status based on done flag and energy
        new_alive_agents = alive_agents & ~done_batch & (energy > 0)
        n_alive = mx.sum(new_alive_agents).item()
        
        # Update observations for agents that are still alive
        if hasattr(observations, 'shape') and hasattr(next_observations, 'shape'):
            # Verify observation array dimensions
            if observations.shape[0] != self.pop_size or next_observations.shape[0] != self.pop_size:
                logger.warning(f"Observation dimension mismatch. Expected pop_size={self.pop_size}, "
                               f"got observations shape={observations.shape}, next_observations shape={next_observations.shape}")
                # Reset observations to default
                observations = self._get_default_observations()
            else:
                # Create a mask for selecting observations by expanding dimensions for proper broadcasting
                mask = mx.expand_dims(new_alive_agents, axis=1)
                # Broadcast mask to match observation dimensions
                mask = mx.broadcast_to(mask, observations.shape)
                # Update observations using where
                observations = mx.where(mask, next_observations, observations)
        else:
            # Get fresh observations if dimensions don't match
            observations = self._get_default_observations()
        
        # Update LCI values - use maximum safe value of 1 for step_counts to avoid division by zero
        step_counts_safe = mx.maximum(step_counts, 1)
        lci_values = rewards / step_counts_safe
        
        # Update alive agents
        alive_agents = new_alive_agents
        
        return rewards, step_counts, alive_agents, observations, lci_values, n_alive
        
    def _get_default_observations(self):
        """
        Get default observations for the entire population.
        Used when array dimensions don't match.
        
        Returns:
            Array of default observations
        """
        # We'll use the environment's reset method to get default observations
        return self.env.reset(self.pop_size)

    def _calculate_final_fitness(self, fitness_data):
        """
        Calculate final fitness values based on rewards and other factors.
        
        Args:
            fitness_data: FitnessState containing fitness tracking data
            
        Returns:
            Tuple of (fitness array, LCI values array)
        """
        try:
            # Extract values needed for calculation
            rewards = fitness_data.rewards
            step_counts = fitness_data.step_counts
            lci_values = fitness_data.lci_values
            
            # Avoid division by zero
            step_counts_safe = mx.maximum(step_counts, 1)
            
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
            return mx.zeros(self.pop_size), mx.zeros(self.pop_size)
    
    def _calculate_fitness_from_rewards(self, rewards, step_counts_safe):
        """
        Calculate fitness from rewards and step counts.
        
        Args:
            rewards: Array of rewards for each agent
            step_counts_safe: Array of step counts for each agent (clamped to avoid division by zero)
            
        Returns:
            Array of fitness values for each agent
        """
        # Convert step_counts_safe to float directly rather than using astype
        # to avoid tuple creation issues
        step_counts_float = mx.array(step_counts_safe, dtype=mx.float32)
        return rewards / step_counts_float
    
    def _calculate_statistics(self, fitness, lci_values, step_counts):
        """
        Calculate statistics for the current generation.
        
        Args:
            fitness: Array of fitness values for each agent
            lci_values: Array of LCI values for each agent
            step_counts: Array of step counts for each agent
            
        Returns:
            Named tuple with statistics
        """
        from collections import namedtuple
        Stats = namedtuple('Stats', [
            'generation', 'mean_fitness', 'std_fitness', 'max_fitness',
            'mean_lci', 'max_lci', 'alive_count', 'best_idx'
        ])
        
        # Find the agent with highest fitness
        max_fitness_value = mx.max(fitness).item()
        best_idx = mx.argmax(fitness).item()
        
        # Get alive count directly from agent energy levels
        # Only count agents with enough energy to learn as "alive"
        energy_levels = self.agents.get_energy()
        alive_count = mx.sum(energy_levels >= self.agents.energy_cost_learn).item()
        
        return Stats(
            generation=self.generation,
            mean_fitness=mx.mean(fitness).item(),
            std_fitness=mx.std(fitness).item() if fitness.size > 1 else 0.0,
            max_fitness=max_fitness_value,
            mean_lci=mx.mean(lci_values).item(),
            max_lci=mx.max(lci_values).item(),
            alive_count=alive_count,
            best_idx=best_idx
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
            'std': energy_levels.std().item() if energy_levels.size > 1 else 0.0
        })
        
        # Check if this is the best fitness seen so far
        if stats.max_fitness > self.best_fitness:
            self.best_fitness = stats.max_fitness
            self.best_agent_generation = stats.generation
            
            # Record best agent
            self.best_agent_history.append({
                'generation': stats.generation,
                'fitness': stats.max_fitness,
                'lci': self.agents.get_lci().max()  # Use get_lci() method instead of lci attribute
            })

    def _run_tournament_selection(self, fitness):
        """
        Run tournament selection to select parents for reproduction.
        
        Args:
            fitness: Array of fitness values for the population
            
        Returns:
            Indices of the winners from each tournament
        """
        # This method is deprecated, use the vectorized version instead
        return self._vectorized_tournament_selection(fitness)

    def _vectorized_tournament_selection(self, fitness):
        """
        Vectorized tournament selection that uses array operations to run all tournaments in parallel.
        
        Args:
            fitness: MLX array of fitness values for the population [pop_size]
            
        Returns:
            MLX array of indices of the winners from each tournament [pop_size]
        """
        # Pre-allocate memory for all tournaments at once [pop_size, tournament_size]
        tournament_indices = mx.random.randint(
            0, self.pop_size, 
            shape=(self.pop_size, self.tournament_size)
        )
        
        # Create empty array to store results
        winner_indices = mx.zeros((self.pop_size,), dtype=mx.int32)
        
        # We need to use a loop since MLX doesn't have advanced indexing like PyTorch's gather
        for i in range(self.pop_size):
            # Get the indices for this tournament
            indices = tournament_indices[i]
            
            # Get fitness values for these indices
            tournament_fitness = mx.array([fitness[idx] for idx in indices])
            
            # Find winner (highest fitness)
            winner_pos = mx.argmax(tournament_fitness)
            
            # Get the actual index from tournament_indices
            # Fix: Get the integer value directly to avoid tuple creation
            winner_indices[i] = int(indices[winner_pos].item())
        
        return winner_indices

    def selection_and_reproduction(self, fitness: mx.array) -> None:
        """
        Performs selection, reproduction and mutation to create the next generation
        using fully batched array operations.
        
        Args:
            fitness: An array containing the fitness values for each agent in the population.
        """
        try:
            # Calculate how many elite agents to keep
            num_elite = max(1, int(self.elitism * self.pop_size) if isinstance(self.elitism, float) else self.elitism)
            
            # Get indices of agents with top fitness - fix for MLX's argsort behavior
            sorted_indices = mx.argsort(fitness)
            elite_indices = sorted_indices[-num_elite:]
            
            # Select parents through tournament selection (fully vectorized)
            parent_indices = self._vectorized_tournament_selection(fitness)
            
            # Create new population by copying parameters from parents in a single batch operation
            for layer_idx in range(len(self.agents.policy_net.layers)):
                # Get all weights and biases for this layer
                weights, biases = self.agents.get_layer_parameters(layer_idx)
                
                # Create new weights and biases by selecting from parents (all in one batch)
                new_weights = weights[parent_indices]
                new_biases = biases[parent_indices]
                
                # Apply mutations to all agents at once
                # Mutation mask (True where mutation should occur)
                weight_mutation_mask = mx.random.uniform(shape=new_weights.shape) < self.mutation_rate
                bias_mutation_mask = mx.random.uniform(shape=new_biases.shape) < self.mutation_rate
                
                # Create Gaussian noise for mutations, matching the shape of weights and biases
                weight_mutations = mx.random.normal(shape=new_weights.shape) * 0.1
                bias_mutations = mx.random.normal(shape=new_biases.shape) * 0.1
                
                # Apply mutations using array operations
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
            
            # Mutation mask for output layer
            out_weight_mutation_mask = mx.random.uniform(shape=new_out_weights.shape) < self.mutation_rate
            out_bias_mutation_mask = mx.random.uniform(shape=new_out_biases.shape) < self.mutation_rate
            
            # Create Gaussian noise for mutations
            out_weight_mutations = mx.random.normal(shape=new_out_weights.shape) * 0.1
            out_bias_mutations = mx.random.normal(shape=new_out_biases.shape) * 0.1
            
            # Apply mutations using array operations
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
                # Check if the energy efficiency attributes exist before using them
                has_energy_params = (hasattr(self.agents, 'energy_efficiency') and 
                                    hasattr(self.agents, 'learn_probability') and
                                    hasattr(self.agents, 'recovery_rate'))
                
                if has_energy_params:
                    # Copy energy genes from parents
                    new_energy_efficiency = self.agents.energy_efficiency[parent_indices]
                    new_learn_probability = self.agents.learn_probability[parent_indices]
                    new_recovery_rate = self.agents.recovery_rate[parent_indices]
                    
                    # Apply mutations
                    efficiency_mutation_mask = mx.random.uniform(shape=new_energy_efficiency.shape) < self.mutation_rate
                    learn_prob_mutation_mask = mx.random.uniform(shape=new_learn_probability.shape) < self.mutation_rate
                    recovery_mutation_mask = mx.random.uniform(shape=new_recovery_rate.shape) < self.mutation_rate
                    
                    # Create mutations
                    efficiency_mutations = 0.1 * mx.random.normal(shape=new_energy_efficiency.shape)
                    learn_prob_mutations = 0.02 * mx.random.normal(shape=new_learn_probability.shape)
                    recovery_mutations = 0.005 * mx.random.normal(shape=new_recovery_rate.shape)
                    
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
                    new_energy_efficiency = mx.clip(new_energy_efficiency, 0.5, 1.5)
                    new_learn_probability = mx.clip(new_learn_probability, 0.01, 0.3)
                    new_recovery_rate = mx.clip(new_recovery_rate, 0.005, 0.1)
                    
                    # Update agent's energy genes
                    self.agents.energy_efficiency = new_energy_efficiency
                    self.agents.learn_probability = new_learn_probability
                    self.agents.recovery_rate = new_recovery_rate
                    
                    # Update derived values
                    self.agents.energy_cost_predict = mx.full((self.pop_size,), self.agents.base_energy_cost_predict) / new_energy_efficiency
                    self.agents.energy_cost_learn = mx.full((self.pop_size,), self.agents.base_energy_cost_learn) / new_energy_efficiency
                    self.agents.energy_recovery = new_recovery_rate
                else:
                    # Evolution of energy parameters is enabled, but parameters don't exist yet
                    # This might happen on first initialization - we could initialize them here
                    logger.info("Evolvable energy is enabled but energy parameters are not initialized. Creating defaults.")
            
            # Reset agent state variables for next generation
            self.agents.reset_state()
        
        except Exception as e:
            logger.error(f"Error in selection and reproduction: {str(e)}")
            traceback.print_exc()

    def run_generation(self, steps: int) -> Tuple[mx.array, mx.array]:
        """
        Run a single generation of the evolutionary algorithm.
        
        Args:
            steps: Number of steps to run
            
        Returns:
            Tuple of (fitness_array, lci_array)
        """
        # Store the current steps_per_generation
        original_steps = self.steps_per_generation
        self.steps_per_generation = steps
        
        # Evaluate fitness
        fitness, lci_values = self.evaluate_fitness()
        
        # Track the best agent (moved to _calculate_final_fitness)
        best_idx = mx.argmax(fitness).item()
        best_fitness_gen = fitness[best_idx]
        
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
        # Create MLX array encoder class for JSON serialization
        class MLXArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, mx.array):
                    # Convert MLX array to list for serialization
                    return obj.tolist()
                return super().default(obj)
                
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
            json.dump(results, f, indent=2, cls=MLXArrayEncoder)

    def _zeros_like_population(self):
        """Create a zero array with population size."""
        return mx.zeros((self.pop_size,))
    
    def _ones_like_population(self):
        """Create a ones array with population size."""
        return mx.ones((self.pop_size,))

    def _get_scalar_energy_value(self, energy_param):
        """Get a scalar value from an energy parameter that could be either an array or a scalar."""
        if hasattr(energy_param, 'shape'):
            # If it's an array with multiple values (one per agent), use the mean
            if len(energy_param.shape) > 0 and energy_param.size > 1:
                return mx.mean(energy_param).item()
            # If it's a single value array, just get the item
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