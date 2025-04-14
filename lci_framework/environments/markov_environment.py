"""
Markov Environment Module

This module implements a Markov process environment with configurable volatility
for studying agent adaptation to changing conditions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logger
logger = logging.getLogger(__name__)

class MarkovEnvironment:
    """
    Markov process environment with configurable volatility
    
    This environment implements a Markov process where transition probabilities
    can change over time according to a volatility schedule.
    
    Attributes:
        n_states (int): Number of possible states in the environment
        sparse_transitions (bool): Whether to use sparse representation for transitions
        current_state (int): The current state of the environment
        transitions (dict or np.ndarray): The transition probabilities between states
        volatility_schedule (list): Schedule of (timesteps, volatility_level) pairs
        current_phase (int): Index of the current phase in the volatility schedule
        current_timestep (int): Current timestep in the simulation
        current_volatility (float): Current volatility level (0 to 1)
        volatility_history (list): History of volatility values for analysis
    """
    
    def __init__(
        self, 
        n_states: int = 20,
        sparse_transitions: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the Markov environment
        
        Args:
            n_states: Number of possible states in the environment
            sparse_transitions: Whether to use sparse representation for transitions
            seed: Random seed for reproducibility
        """
        self.n_states = n_states
        self.sparse_transitions = sparse_transitions
        self.current_state = 0
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Initialized random seed: {seed}")
            
        # Initialize transition matrix (to be filled with actual probabilities)
        self._initialize_transition_matrix()
        
        # Default volatility schedule: (timesteps, volatility_level)
        # where volatility_level is between 0 (stable) and 1 (highly volatile)
        self.volatility_schedule = [
            (1000, 0.0),   # Initial stable phase
            (1000, 0.5),   # Medium volatility phase
            (1000, 0.0),   # Return to stability
            (1000, 0.8),   # High volatility phase
            (1000, 0.0)    # Final stable phase
        ]
        self.current_phase = 0
        self.current_timestep = 0
        self.current_volatility = self.volatility_schedule[0][1]
        
        # Tracking history for analysis
        self.volatility_history = []
        
        logger.debug(f"Created MarkovEnvironment with {n_states} states")
        
    def _initialize_transition_matrix(self):
        """
        Initialize transition probabilities with some structure
        
        For sparse transitions, uses a dictionary to store only non-zero transitions.
        For dense transitions, uses a full matrix.
        """
        if self.sparse_transitions:
            # For memory efficiency, use dictionary to store only non-zero transitions
            self.transitions = {}
            for s in range(self.n_states):
                # Each state connects to a limited number of next states
                next_states = np.random.choice(
                    self.n_states, 
                    size=min(5, self.n_states), 
                    replace=False
                )
                # Initialize with random probabilities
                probs = np.random.dirichlet(np.ones(len(next_states)))
                self.transitions[s] = list(zip(next_states, probs))
        else:
            # Full matrix implementation
            self.transitions = np.zeros((self.n_states, self.n_states))
            for s in range(self.n_states):
                # For each state, create random transition probabilities
                row = np.zeros(self.n_states)
                # Choose a subset of possible next states
                next_states = np.random.choice(
                    self.n_states, 
                    size=min(5, self.n_states), 
                    replace=False
                )
                # Set probabilities for those states
                row[next_states] = np.random.dirichlet(np.ones(len(next_states)))
                self.transitions[s] = row
                
        logger.debug("Initialized transition matrix")
    
    def update_environment(self, timestep: int):
        """
        Update environment based on the current volatility schedule
        
        Args:
            timestep: Current simulation timestep
        """
        self.current_timestep = timestep
        
        # Check if we need to move to the next phase
        current_phase_length = self.volatility_schedule[self.current_phase][0]
        phase_step = timestep % current_phase_length
        
        if phase_step == 0 and timestep > 0:
            self.current_phase = (self.current_phase + 1) % len(self.volatility_schedule)
            logger.info(f"Moving to volatility phase {self.current_phase}")
        
        # Get current volatility level
        self.current_volatility = self.volatility_schedule[self.current_phase][1]
        self.volatility_history.append(self.current_volatility)
        
        # Apply volatility - modify transition matrix based on volatility level
        self._apply_volatility(self.current_volatility)
    
    def _apply_volatility(self, volatility: float):
        """
        Modify transition probabilities based on volatility level
        
        Args:
            volatility: Level of volatility between 0 and 1
        """
        if volatility <= 0:
            return  # No changes needed
        
        if self.sparse_transitions:
            for s in range(self.n_states):
                if np.random.random() < volatility:
                    # With probability = volatility, change transitions for this state
                    next_states = np.random.choice(
                        self.n_states, 
                        size=min(5, self.n_states), 
                        replace=False
                    )
                    probs = np.random.dirichlet(np.ones(len(next_states)))
                    self.transitions[s] = list(zip(next_states, probs))
        else:
            # For each row that we choose to modify
            for s in range(self.n_states):
                if np.random.random() < volatility:
                    # Reset row
                    row = np.zeros(self.n_states)
                    # Choose a subset of possible next states
                    next_states = np.random.choice(
                        self.n_states, 
                        size=min(5, self.n_states), 
                        replace=False
                    )
                    # Set probabilities for those states
                    row[next_states] = np.random.dirichlet(np.ones(len(next_states)))
                    self.transitions[s] = row
    
    def step(self) -> Tuple[int, float]:
        """
        Take one step in the environment
        
        Returns:
            Tuple of (next_state, reward)
        """
        # Get next state based on transition probabilities
        if self.sparse_transitions:
            next_states, probs = zip(*self.transitions[self.current_state])
            next_state = np.random.choice(next_states, p=probs)
        else:
            next_state = np.random.choice(
                self.n_states, 
                p=self.transitions[self.current_state]
            )
        
        # Calculate reward (in this simple case, reward is the state number)
        # This is a placeholder - customize based on your specific problem
        reward = next_state / self.n_states
        
        # Update current state
        self.current_state = next_state
        
        return next_state, reward
    
    def reset(self, initial_state: Optional[int] = None) -> int:
        """
        Reset environment to initial state
        
        Args:
            initial_state: Optional specific state to reset to
            
        Returns:
            The initial state after reset
        """
        if initial_state is not None and 0 <= initial_state < self.n_states:
            self.current_state = initial_state
        else:
            self.current_state = np.random.choice(self.n_states)
        
        # Reset tracking
        self.current_timestep = 0
        self.current_phase = 0
        self.volatility_history = []
        
        # Re-initialize transition matrix
        self._initialize_transition_matrix()
        
        logger.debug(f"Environment reset to state {self.current_state}")
        return self.current_state
    
    def get_observation(self, state: Optional[int] = None) -> np.ndarray:
        """
        Get observation vector from state
        
        Args:
            state: State to get observation for (default: current state)
            
        Returns:
            Observation vector (one-hot encoding of state)
        """
        if state is None:
            state = self.current_state
        
        # Convert to one-hot encoding
        obs = np.zeros(self.n_states)
        obs[state] = 1.0
        
        return obs
        
    def set_volatility_schedule(self, schedule: List[Tuple[int, float]]):
        """
        Set a custom volatility schedule
        
        Args:
            schedule: List of (timesteps, volatility_level) tuples
        """
        if not schedule:
            logger.warning("Empty volatility schedule provided, using default")
            return
            
        self.volatility_schedule = schedule
        self.current_phase = 0
        self.current_volatility = schedule[0][1]
        logger.info(f"Set custom volatility schedule with {len(schedule)} phases") 