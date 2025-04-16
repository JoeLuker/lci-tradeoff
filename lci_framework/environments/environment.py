import mlx.core as mx
import logging
from typing import Tuple, Optional, Dict, List, Any
logger = logging.getLogger(__name__)

class VectorizedMarkovEnvironment:
    """
    A vectorized implementation of a Markov environment that processes multiple
    states in parallel using tensor operations.
    
    The environment consists of states and transition probabilities, with rewards
    for each state-action pair.
    """
    
    def __init__(self, 
                 n_states: int, 
                 n_actions: int,
                 volatility: float = 0.1,
                 reward_noise: float = 0.1):
        """
        Initialize a vectorized Markov environment.
        
        Args:
            n_states: Number of possible states
            n_actions: Number of possible actions
            volatility: Rate of change in transition probabilities
            reward_noise: Standard deviation of noise added to rewards
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.volatility = volatility
        self.reward_noise = reward_noise
        
        # Create transition matrix: shape [n_states, n_actions, n_states]
        # This represents P(s'|s,a) - the probability of transitioning to state s'
        # given current state s and action a
        self.transition_matrix = self._create_transition_matrix()
        
        # Create reward matrix: shape [n_states, n_actions]
        # This represents the reward for taking action a in state s
        self.reward_matrix = mx.random.uniform(shape=(n_states, n_actions))
        
        # Current state for each agent in the population
        # Will be initialized when reset is called with a population size
        self.current_states = None
        self.pop_size = None
        
    def _create_transition_matrix(self) -> mx.array:
        """
        Create a random transition matrix where each row sums to 1.
        
        Returns:
            Array of shape [n_states, n_actions, n_states]
        """
        # Create random values
        transitions = mx.random.uniform(shape=(self.n_states, self.n_actions, self.n_states))
        
        # Normalize across destination states to create probabilities
        return transitions / mx.sum(transitions, axis=2, keepdims=True)
    
    def update_transitions(self) -> None:
        """
        Update transition probabilities based on volatility.
        """
        # Generate random perturbations
        noise = mx.random.normal(shape=(self.n_states, self.n_actions, self.n_states)) * self.volatility
        
        # Apply perturbations
        self.transition_matrix = self.transition_matrix + noise
        
        # Ensure values are positive
        self.transition_matrix = mx.maximum(self.transition_matrix, 1e-6)
        
        # Renormalize to ensure rows sum to 1
        self.transition_matrix = self.transition_matrix / mx.sum(self.transition_matrix, axis=2, keepdims=True)
    
    def update_rewards(self) -> None:
        """
        Update rewards with random noise.
        """
        noise = mx.random.normal(shape=(self.n_states, self.n_actions)) * self.reward_noise
        self.reward_matrix = mx.clip(self.reward_matrix + noise, 0.0, 1.0)
    
    def reset(self, pop_size: int) -> mx.array:
        """
        Reset the environment for a population of agents.
        
        Args:
            pop_size: Number of agents in the population
            
        Returns:
            Array of initial observations (one-hot encoded states) 
            with shape [pop_size, n_states]
        """
        self.pop_size = pop_size
        
        # Randomly assign initial states
        self.current_states = mx.random.randint(0, self.n_states, shape=(pop_size,))
        
        # Return one-hot encoded states as observations
        return self._get_observations()
    
    def _get_observations(self) -> mx.array:
        """
        Convert current states to one-hot encoded observations.
        
        Returns:
            Array of one-hot encoded states with shape [pop_size, n_states]
        """
        # Create a range of state indices
        state_indices = mx.arange(self.n_states)
        
        # Reshape current_states to allow broadcasting: [pop_size, 1]
        states_reshaped = mx.reshape(self.current_states, (self.pop_size, 1))
        
        # Compare each state index with current_states using broadcasting
        # This creates a boolean array of shape [pop_size, n_states] where
        # True indicates a match (the state is active)
        one_hot_bool = mx.equal(states_reshaped, state_indices)
        
        # Convert boolean array to float
        return mx.array(one_hot_bool, dtype=mx.float32)
    
    def step(self, actions: mx.array) -> Tuple[mx.array, mx.array, mx.array, Dict]:
        """
        Take one step in the environment for all agents based on their actions.
        
        Args:
            actions: Array of action indices with shape [pop_size]
                    or action probabilities with shape [pop_size, n_actions]
            
        Returns:
            observations: New observations (one-hot encoded states) [pop_size, n_states]
            rewards: Rewards for each agent [pop_size]
            done: Boolean array indicating if episodes are done [pop_size]
            info: Additional information dictionary
        """
        # Convert action probabilities to indices if needed
        if actions.ndim > 1:
            actions = mx.argmax(actions, axis=1)
        
        # Ensure actions are valid
        actions = mx.clip(actions, 0, self.n_actions - 1)
        
        # Ensure current states array has the right shape
        if self.current_states.shape[0] != self.pop_size:
            # If we have the wrong number of states, reinitialize
            self.current_states = mx.random.randint(0, self.n_states, shape=(self.pop_size,))
            logger.warning(f"Fixed state dimensions mismatch: reset to {self.pop_size} states")
        
        # Get transition probabilities for current states and chosen actions
        transition_probs = self.transition_matrix[self.current_states, actions]
        
        # Sample next states based on transition probabilities - properly vectorized
        log_probs = mx.log(mx.maximum(transition_probs, 1e-10))  # Avoid log(0)
        next_states = mx.random.categorical(log_probs)
        
        # Get rewards for current states and chosen actions
        rewards = self.reward_matrix[self.current_states, actions]
        
        # Add some noise to rewards
        reward_noise = mx.random.normal(shape=rewards.shape) * self.reward_noise
        rewards = mx.clip(rewards + reward_noise, 0.0, 1.0)
        
        # Update current states
        self.current_states = next_states
        
        # Get new observations
        observations = self._get_observations()
        
        # In this environment, episodes don't end
        done = mx.zeros((self.pop_size,), dtype=mx.bool_)
        
        # Additional information
        info = {}
        
        return observations, rewards, done, info
    
    def update_environment(self) -> None:
        """
        Update environment dynamics (transition probabilities and rewards).
        """
        self.update_transitions()
        self.update_rewards()
    
    def sample_random_actions(self) -> mx.array:
        """
        Sample random action indices.
        
        Returns:
            Array of random action indices with shape [pop_size]
        """
        return mx.random.randint(0, self.n_actions, shape=(self.pop_size,))
    
    def sample_random_action_probs(self) -> mx.array:
        """
        Sample random action probabilities.
        
        Returns:
            Array of random action probabilities with shape [pop_size, n_actions]
        """
        probs = mx.random.uniform(shape=(self.pop_size, self.n_actions))
        return probs / mx.sum(probs, axis=1, keepdims=True)
    
    def get_n_states(self) -> int:
        """
        Get the number of states in the environment.
        
        Returns:
            Number of states
        """
        return self.n_states
    
    def get_n_actions(self) -> int:
        """
        Get the number of actions in the environment.
        
        Returns:
            Number of actions
        """
        return self.n_actions