"""
LCI Agent Module

This module implements the LCI (Losslessness, Compression, Invariance) agent.
The agent uses a neural network model to balance these three fundamental properties.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import logging
import random
from abc import ABC, abstractmethod

# Configure logger
logger = logging.getLogger(__name__)

class NeuralModel(nn.Module):
    """
    Neural network model for the LCI agent
    
    Implements a configurable feedforward neural network.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        n_layers: int, 
        dropout_rate: float = 0.0
    ):
        """
        Initialize neural network model
        
        Args:
            input_size: Dimension of input features
            hidden_size: Size of hidden layers
            n_layers: Number of hidden layers
            dropout_rate: Dropout probability (0 to 1)
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, input_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class BaseAgent(ABC):
    """Abstract base class for agents"""
    
    @abstractmethod
    def predict(self, observation):
        """Predict next state based on current observation"""
        pass
        
    @abstractmethod
    def learn(self, obs, next_obs):
        """Update model based on observed transition"""
        pass
        
    @abstractmethod
    def receive_reward(self, reward):
        """Process received reward"""
        pass
        
    @abstractmethod
    def is_alive(self):
        """Check if agent has enough energy to continue"""
        pass
        
    @abstractmethod
    def get_state(self):
        """Get agent's current state for analysis"""
        pass


class LCIAgent(BaseAgent):
    """
    Neural network-based agent with configurable L/C/I parameters
    
    This agent implements a neural network that balances:
    - L (Losslessness): Controlled by learning rate and model capacity
    - C (Compression): Controlled by network size
    - I (Invariance): Controlled by regularization and architecture choices
    
    Attributes:
        input_size (int): Size of the observation vector
        learning_rate (float): Learning rate for model updates (L parameter)
        hidden_size (int): Size of hidden layers (C parameter)
        n_layers (int): Number of hidden layers (C parameter)
        l1_reg (float): L1 regularization strength (I parameter)
        l2_reg (float): L2 regularization strength (I parameter)
        dropout_rate (float): Dropout rate for improving invariance (I parameter)
        energy (float): Current energy budget
        initial_energy (float): Starting energy budget
        agent_id (int): Unique identifier for the agent
        model (nn.Module): Neural network model
        lci_params (dict): Dictionary of L/C/I parameters
    """
    
    def __init__(
        self,
        input_size: int,
        # L parameters
        learning_rate: float = 0.01,  # Higher = higher L
        # C parameters
        hidden_size: int = 16,        # Larger = lower C
        n_layers: int = 1,            # More = lower C
        # I parameters
        l1_reg: float = 0.0,          # Higher = higher I
        l2_reg: float = 0.001,        # Higher = higher I
        dropout_rate: float = 0.0,    # Higher = higher I
        # Energy budget
        initial_energy: float = 1000.0,
        # Agent ID for tracking
        agent_id: Optional[int] = None,
        # Random seed
        seed: Optional[int] = None
    ):
        """
        Initialize the LCI agent
        
        Args:
            input_size: Size of the observation vector
            learning_rate: Learning rate for model updates
            hidden_size: Size of hidden layers
            n_layers: Number of hidden layers
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            dropout_rate: Dropout rate for improving invariance
            initial_energy: Starting energy budget
            agent_id: Unique identifier for the agent
            seed: Random seed for reproducibility
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.agent_id = agent_id if agent_id is not None else id(self)
        
        # Input validation
        if input_size <= 0:
            raise ValueError("Input size must be positive")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
        if n_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if not 0 <= l1_reg <= 1:
            raise ValueError("L1 regularization must be between 0 and 1")
        if not 0 <= l2_reg <= 1:
            raise ValueError("L2 regularization must be between 0 and 1")
        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"Set random seed: {seed}")
            
        # Initialize LCI parameters for tracking and analysis
        self.lci_params = {
            "L": learning_rate,
            "C": 1.0 / (hidden_size * n_layers),  # Inverse of model size
            "I": (l1_reg + l2_reg + dropout_rate) / 3.0  # Average of I-promoting techniques
        }
        
        # Build the model
        self.model = NeuralModel(input_size, hidden_size, n_layers, dropout_rate)
        
        # Setup optimizer with weight decay for L2 regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # History for analysis
        self.reward_history = []
        self.energy_history = []
        self.prediction_error_history = []
        
        logger.debug(f"Created LCIAgent {self.agent_id} with LCI balance: {self.get_lci_balance():.3f}")
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict the next state based on current observation
        
        Args:
            observation: Current state observation
            
        Returns:
            Predicted next state
        """
        if not self.is_alive():
            logger.warning(f"Agent {self.agent_id} attempted to predict with no energy")
            return 0
            
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation)
        
        # Forward pass through model
        with torch.no_grad():
            prediction = self.model(obs_tensor)
        
        # Calculate energy cost of inference
        inference_cost = self._calculate_inference_cost()
        self.energy -= inference_cost
        self.energy_history.append(self.energy)
        
        # Convert prediction to state index
        pred_state = torch.argmax(prediction).item()
        
        return pred_state
    
    def learn(self, obs: np.ndarray, next_obs: np.ndarray) -> float:
        """
        Update model based on observed transition
        
        Args:
            obs: Current observation
            next_obs: Next observation
            
        Returns:
            Loss value
        """
        # Check if we have enough energy for learning
        learning_cost = self._calculate_learning_cost()
        if self.energy < learning_cost:
            logger.warning(f"Agent {self.agent_id} attempted to learn with insufficient energy")
            return 0.0  # Can't learn, not enough energy
            
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs)
        next_obs_tensor = torch.FloatTensor(next_obs)
        
        # Forward pass
        prediction = self.model(obs_tensor)
        
        # Calculate loss
        loss = self.criterion(prediction, next_obs_tensor)
        
        # Add L1 regularization if needed
        if self.l1_reg > 0:
            l1_loss = 0
            for param in self.model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += self.l1_reg * l1_loss
            
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Deduct energy cost
        self.energy -= learning_cost
        self.energy_history.append(self.energy)
        
        # Track error for analysis
        self.prediction_error_history.append(loss.item())
        
        return loss.item()
    
    def receive_reward(self, reward: float):
        """
        Process received reward
        
        Args:
            reward: Reward value
        """
        self.reward_history.append(reward)
        
        # Convert reward to energy (simple direct conversion for now)
        energy_gain = reward * 10  # Scale factor
        self.energy += energy_gain
        self.energy_history.append(self.energy)
        
        logger.debug(f"Agent {self.agent_id} received reward {reward:.3f}, energy: {self.energy:.1f}")
    
    def _calculate_inference_cost(self) -> float:
        """
        Calculate energy cost of running inference
        
        Returns:
            Energy cost
        """
        # Simple model: cost proportional to model size
        return 0.01 * self.hidden_size * self.n_layers
    
    def _calculate_learning_cost(self) -> float:
        """
        Calculate energy cost of a learning step
        
        Returns:
            Energy cost
        """
        # Learning is more expensive than inference
        # Cost increases with learning rate (higher L = higher cost)
        base_cost = 0.1 * self.hidden_size * self.n_layers
        l_factor = 1.0 + 2.0 * self.learning_rate  # Learning rate impacts cost
        return base_cost * l_factor
    
    def get_state(self) -> Dict:
        """
        Get agent's current state for analysis
        
        Returns:
            Dictionary with agent state information
        """
        return {
            "id": self.agent_id,
            "energy": self.energy,
            "lci_params": self.lci_params,
            "avg_reward": np.mean(self.reward_history[-100:]) if self.reward_history else 0,
            "avg_error": np.mean(self.prediction_error_history[-100:]) if self.prediction_error_history else 0
        }
    
    def is_alive(self) -> bool:
        """
        Check if agent has enough energy to continue
        
        Returns:
            True if agent has positive energy, False otherwise
        """
        return self.energy > 0
    
    def get_lci_balance(self) -> float:
        """
        Calculate LCI balance using the formula: 3*L*C*I / (L^3 + C^3 + I^3)
        
        This formula reaches its maximum when L=C=I, representing perfect balance.
        
        Returns:
            LCI balance score (higher = more balanced, range 0-1)
        """
        L = self.lci_params["L"]
        C = self.lci_params["C"]
        I = self.lci_params["I"]
        
        numerator = 3 * L * C * I
        denominator = L**3 + C**3 + I**3
        
        # Avoid division by zero
        if denominator == 0:
            return 0
            
        return numerator / denominator
        
    def save_model(self, filepath: str):
        """
        Save the agent's model to a file
        
        Args:
            filepath: Path to save the model
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lci_params': self.lci_params,
                'agent_id': self.agent_id,
                'energy': self.energy
            }, filepath)
            logger.info(f"Saved agent {self.agent_id} model to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
    def load_model(self, filepath: str):
        """
        Load the agent's model from a file
        
        Args:
            filepath: Path to load the model from
        """
        try:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lci_params = checkpoint['lci_params']
            self.agent_id = checkpoint['agent_id']
            self.energy = checkpoint['energy']
            logger.info(f"Loaded agent {self.agent_id} model from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}") 