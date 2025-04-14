import torch
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List

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
                 reward_noise: float = 0.1,
                 device: Optional[torch.device] = None):
        """
        Initialize a vectorized Markov environment.
        
        Args:
            n_states: Number of possible states
            n_actions: Number of possible actions
            volatility: Rate of change in transition probabilities
            reward_noise: Standard deviation of noise added to rewards
            device: Device for computation (defaults to MPS or CUDA)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.volatility = volatility
        self.reward_noise = reward_noise
        
        # Set up device (MPS or CUDA only)
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device for vectorized environment")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device for vectorized environment")
        else:
            raise RuntimeError("No GPU available. This implementation requires GPU acceleration.")
        
        # Create transition matrix: shape [n_states, n_actions, n_states]
        # This represents P(s'|s,a) - the probability of transitioning to state s'
        # given current state s and action a
        self.transition_matrix = self._create_transition_matrix()
        
        # Create reward matrix: shape [n_states, n_actions]
        # This represents the reward for taking action a in state s
        self.reward_matrix = torch.rand(n_states, n_actions, device=self.device)
        
        # Current state for each agent in the population
        # Will be initialized when reset is called with a population size
        self.current_states = None
        self.pop_size = None
        
    def _create_transition_matrix(self) -> torch.Tensor:
        """
        Create a random transition matrix where each row sums to 1.
        
        Returns:
            Tensor of shape [n_states, n_actions, n_states]
        """
        # Create random values
        transitions = torch.rand(self.n_states, self.n_actions, self.n_states, device=self.device)
        
        # Normalize across destination states to create probabilities
        return transitions / transitions.sum(dim=2, keepdim=True)
    
    def update_transitions(self) -> None:
        """
        Update transition probabilities based on volatility.
        """
        # Generate random perturbations
        noise = torch.randn(self.n_states, self.n_actions, self.n_states, device=self.device) * self.volatility
        
        # Apply perturbations
        self.transition_matrix = self.transition_matrix + noise
        
        # Ensure values are positive
        self.transition_matrix = torch.clamp(self.transition_matrix, min=1e-6)
        
        # Renormalize to ensure rows sum to 1
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(dim=2, keepdim=True)
    
    def update_rewards(self) -> None:
        """
        Update rewards with random noise.
        """
        noise = torch.randn(self.n_states, self.n_actions, device=self.device) * self.reward_noise
        self.reward_matrix = torch.clamp(self.reward_matrix + noise, min=0.0, max=1.0)
    
    def reset(self, pop_size: int) -> torch.Tensor:
        """
        Reset the environment for a population of agents.
        
        Args:
            pop_size: Number of agents in the population
            
        Returns:
            Tensor of initial observations (one-hot encoded states) 
            with shape [pop_size, n_states]
        """
        self.pop_size = pop_size
        
        # Randomly assign initial states
        self.current_states = torch.randint(0, self.n_states, (pop_size,), device=self.device)
        
        # Return one-hot encoded states as observations
        return self._get_observations()
    
    def _get_observations(self) -> torch.Tensor:
        """
        Convert current states to one-hot encoded observations.
        
        Returns:
            Tensor of one-hot encoded states with shape [pop_size, n_states]
        """
        # Create one-hot encoded tensor
        observations = torch.zeros(self.pop_size, self.n_states, device=self.device)
        observations.scatter_(1, self.current_states.unsqueeze(1), 1.0)
        return observations
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Take one step in the environment for all agents based on their actions.
        
        Args:
            actions: Tensor of action indices with shape [pop_size]
                    or action probabilities with shape [pop_size, n_actions]
            
        Returns:
            observations: New observations (one-hot encoded states) [pop_size, n_states]
            rewards: Rewards for each agent [pop_size]
            done: Boolean tensor indicating if episodes are done [pop_size]
            info: Additional information dictionary
        """
        # Convert action probabilities to indices if needed
        if actions.dim() > 1:
            actions = actions.argmax(dim=1)
        
        # Get transition probabilities for current states and chosen actions
        # Shape: [pop_size, n_states]
        transition_probs = torch.stack(
            [self.transition_matrix[self.current_states[i], actions[i]] for i in range(self.pop_size)]
        )
        
        # Sample next states based on transition probabilities
        next_states = torch.zeros(self.pop_size, dtype=torch.long, device=self.device)
        
        # Sample from multinomial distribution for each agent
        for i in range(self.pop_size):
            next_states[i] = torch.multinomial(transition_probs[i], 1).item()
        
        # Get rewards for current states and chosen actions
        rewards = self.reward_matrix[self.current_states, actions]
        
        # Update current states
        self.current_states = next_states
        
        # Get new observations
        observations = self._get_observations()
        
        # In this environment, episodes don't end
        done = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        
        # Additional information
        info = {}
        
        return observations, rewards, done, info
    
    def step_batch(self, action_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Take one step in the environment for all agents based on their action probabilities.
        
        Args:
            action_probs: Tensor of action probabilities with shape [pop_size, n_actions]
            
        Returns:
            observations: New observations (one-hot encoded states) [pop_size, n_states]
            rewards: Rewards for each agent [pop_size]
            done: Boolean tensor indicating if episodes are done [pop_size]
            info: Additional information dictionary
        """
        # Convert to action indices if needed
        if action_probs.dim() > 1:
            actions = action_probs.argmax(dim=1)
        else:
            actions = action_probs
        
        # Get transition probabilities for current states and chosen actions
        next_states = torch.zeros_like(self.current_states)
        rewards = torch.zeros(self.pop_size, device=self.device)
        
        # Process each agent individually to avoid tensor shape issues
        for i in range(self.pop_size):
            # Get transition probabilities for this agent
            trans_probs = self.transition_matrix[self.current_states[i], actions[i]]
            
            # Ensure valid probabilities
            if trans_probs.sum() <= 0:
                # If probabilities are invalid, choose a random next state
                next_states[i] = torch.randint(0, self.n_states, (1,), device=self.device)
            else:
                # Ensure probabilities sum to 1
                trans_probs = torch.clamp(trans_probs, min=1e-6)
                trans_probs = trans_probs / trans_probs.sum()
                
                # Sample next state
                next_states[i] = torch.multinomial(trans_probs, 1).item()
            
            # Get reward
            rewards[i] = self.reward_matrix[self.current_states[i], actions[i]]
        
        # Update current states
        self.current_states = next_states
        
        # Get new observations
        observations = self._get_observations()
        
        # In this environment, episodes don't end
        done = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        
        # Additional information
        info = {}
        
        return observations, rewards, done, info
    
    def update_environment(self) -> None:
        """
        Update the environment dynamics by changing transition probabilities
        and rewards.
        """
        self.update_transitions()
        self.update_rewards()
        
    def sample_random_actions(self) -> torch.Tensor:
        """
        Sample random actions for all agents.
        
        Returns:
            Tensor of random action indices with shape [pop_size]
        """
        return torch.randint(0, self.n_actions, (self.pop_size,), device=self.device)
    
    def sample_random_action_probs(self) -> torch.Tensor:
        """
        Sample random action probabilities for all agents.
        
        Returns:
            Tensor of random action probabilities with shape [pop_size, n_actions]
        """
        probs = torch.rand(self.pop_size, self.n_actions, device=self.device)
        return probs / probs.sum(dim=1, keepdim=True)
    
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