import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class PopulationLinear(nn.Module):
    """
    Linear layer that maintains separate weights for each agent in the population.
    This allows for efficient parallel processing.
    """
    def __init__(self, 
                 pop_size: int, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()
        
        # Set up device
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.warning("Using CPU for tensor operations, which may be slow")
        
        # Initialize weights for the entire population
        # Shape: [pop_size, out_features, in_features]
        self.weight = nn.Parameter(
            torch.randn(pop_size, out_features, in_features, device=self.device) * 0.02
        )
        
        if bias:
            # Shape: [pop_size, out_features]
            self.bias = nn.Parameter(
                torch.zeros(pop_size, out_features, device=self.device)
            )
        else:
            self.register_parameter('bias', None)
        
        self.in_features = in_features
        self.out_features = out_features
        self.pop_size = pop_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the population linear layer.
        
        Args:
            x: Input tensor with shape [pop_size, batch_size, in_features]
            
        Returns:
            Output tensor with shape [pop_size, batch_size, out_features]
        """
        # Make sure x is on the right device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Handle different input shapes
        if x.dim() == 2:  # [pop_size, in_features] - single sample case
            # Reshape for bmm: [pop_size, 1, in_features]
            x = x.unsqueeze(1)
            
            # Batched matrix multiplication: [pop_size, 1, in_features] @ [pop_size, in_features, out_features]
            output = torch.bmm(x, self.weight.transpose(1, 2))
            
            # Reshape to remove singleton dimension: [pop_size, out_features]
            output = output.squeeze(1)
            
            # Add bias
            if self.bias is not None:
                output = output + self.bias
                
        elif x.dim() == 3:  # [pop_size, batch_size, in_features] - batch case
            # Batched matrix multiplication
            output = torch.matmul(x, self.weight.transpose(1, 2))
            
            # Add bias
            if self.bias is not None:
                output = output + self.bias.unsqueeze(1)
                
        else:
            raise ValueError(f"Expected input tensor with 2 or 3 dimensions, got {x.dim()}")
            
        return output

class NeuralPopulation(nn.Module):
    """
    A population of neural networks with shared architecture but individual weights.
    This allows for efficient parallel processing of multiple models.
    """
    
    def __init__(self, 
                 pop_size: int, 
                 input_size: int, 
                 hidden_size: int = 64, 
                 n_layers: int = 2,
                 dropout_rate: float = 0.1,
                 device: Optional[torch.device] = None):
        """
        Initialize a population of neural networks.
        
        Args:
            pop_size: Number of neural networks in the population
            input_size: Size of the input features
            hidden_size: Size of the hidden layers
            n_layers: Number of hidden layers
            dropout_rate: Dropout probability
            device: Device for computation (defaults to MPS or CUDA)
        """
        super().__init__()
        
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # Set up device (MPS or CUDA only)
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device for neural population")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device for neural population")
        else:
            raise RuntimeError("No GPU available. This implementation requires GPU acceleration.")
        
        # Create layers
        self.layers = nn.ModuleList([
            PopulationLinear(
                pop_size=pop_size, 
                in_features=input_size if i == 0 else hidden_size, 
                out_features=hidden_size,
                device=self.device
            )
            for i in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = PopulationLinear(
            pop_size=pop_size, 
            in_features=hidden_size, 
            out_features=1,  # Single output per network
            device=self.device
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Move model to the device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the population of neural networks.
        
        Handles both single samples and batches:
        - For a single sample: x.shape = [pop_size, input_size]
        - For a batch: x.shape = [batch_size, pop_size, input_size]
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with shape [pop_size, 1] or [batch_size, pop_size, 1]
        """
        # Check if input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Forward pass through hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Forward pass through output layer
        output = self.output_layer(x)
        
        return output
    
    def l1_loss(self) -> torch.Tensor:
        """
        Calculate L1 regularization loss for all parameters.
        
        Returns:
            L1 regularization loss
        """
        l1_reg = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg += param.abs().sum()
        return l1_reg
    
    def l2_loss(self) -> torch.Tensor:
        """
        Calculate L2 regularization loss for all parameters.
        
        Returns:
            L2 regularization loss
        """
        l2_reg = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += (param ** 2).sum()
        return l2_reg
    
    def save_population(self, path: str) -> None:
        """
        Save the population state.
        
        Args:
            path: Path to save the population state
        """
        state_dict = self.state_dict()
        torch.save({
            'state_dict': state_dict,
            'pop_size': self.pop_size,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate
        }, path)
    
    def load_population(self, path: str) -> None:
        """
        Load the population state.
        
        Args:
            path: Path to load the population state from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.pop_size = checkpoint['pop_size']
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.n_layers = checkpoint['n_layers']
        self.dropout_rate = checkpoint['dropout_rate']

class TensorLCIAgent:
    """
    Tensor-based implementation of the LCI agent that manages a population
    of agents that can be processed in parallel using tensor operations.
    """
    
    def __init__(self, 
                 pop_size: int,
                 input_size: int,
                 energy_cost_predict: float = 0.01,
                 energy_cost_learn: float = 0.1,
                 energy_init: float = 1.0,
                 energy_recovery: float = 0.05,
                 learning_rate: float = 0.01,
                 hidden_size: int = 64,
                 n_layers: int = 2,
                 dropout_rate: float = 0.1,
                 l1_reg: float = 0.001,
                 l2_reg: float = 0.001,
                 learn_prob: float = 0.1,
                 evolvable_energy: bool = True,
                 device: Optional[torch.device] = None):
        """
        Initialize the tensor LCI agent.
        
        Args:
            pop_size: Number of agents in the population
            input_size: Size of the input features
            energy_cost_predict: Energy cost for making a prediction
            energy_cost_learn: Energy cost for learning
            energy_init: Initial energy level
            energy_recovery: Energy recovery rate
            learning_rate: Learning rate for neural network updates
            hidden_size: Size of the hidden layers
            n_layers: Number of hidden layers
            dropout_rate: Dropout probability
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            learn_prob: Probability of learning on each step
            evolvable_energy: Whether to use evolvable energy parameters
            device: Device for computation (defaults to MPS or CUDA)
        """
        # Set up device (MPS or CUDA only)
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device for LCI agents")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device for LCI agents")
        else:
            raise RuntimeError("No GPU available. This implementation requires GPU acceleration.")
        
        self.pop_size = pop_size
        self.input_size = input_size
        self.evolvable_energy = evolvable_energy
        
        # Fixed energy parameters (used as base values)
        self.base_energy_cost_predict = energy_cost_predict
        self.base_energy_cost_learn = energy_cost_learn
        self.base_energy_init = energy_init
        self.base_energy_recovery = energy_recovery
        self.base_learn_prob = learn_prob
        
        # Create evolvable energy genes for each agent
        if evolvable_energy:
            # Initialize with small random variations around base values
            # Each agent gets its own energy parameters
            self.energy_efficiency = torch.ones(pop_size, device=self.device) + 0.1 * torch.randn(pop_size, device=self.device)
            self.learn_probability = torch.full((pop_size,), learn_prob, device=self.device) + 0.02 * torch.randn(pop_size, device=self.device)
            self.recovery_rate = torch.full((pop_size,), energy_recovery, device=self.device) + 0.005 * torch.randn(pop_size, device=self.device)
            
            # Ensure valid values
            self.energy_efficiency = torch.clamp(self.energy_efficiency, min=0.5, max=1.5)
            self.learn_probability = torch.clamp(self.learn_probability, min=0.01, max=0.3)
            self.recovery_rate = torch.clamp(self.recovery_rate, min=0.005, max=0.1)
            
            # Set up energy parameters based on genes
            self.energy_cost_predict = torch.full((pop_size,), energy_cost_predict, device=self.device) / self.energy_efficiency
            self.energy_cost_learn = torch.full((pop_size,), energy_cost_learn, device=self.device) / self.energy_efficiency
            self.energy_recovery = self.recovery_rate
        else:
            # Fixed parameters for all agents (backward compatibility)
            self.energy_cost_predict = energy_cost_predict
            self.energy_cost_learn = energy_cost_learn
            self.energy_init = energy_init
            self.energy_recovery = energy_recovery
            self.learn_probability = learn_prob
        
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # Initialize neural population
        self.model = NeuralPopulation(
            pop_size=pop_size,
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            device=self.device
        )
        
        # Initialize energy levels for all agents
        self.energy = torch.full((pop_size,), energy_init, device=self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Tracking variables
        self.step_count = torch.zeros(pop_size, device=self.device)
        self.warned_low_energy = torch.zeros(pop_size, dtype=torch.bool, device=self.device)
        
    def predict(self, observation: Union[torch.Tensor, np.ndarray], alive_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make a prediction based on observation.
        
        Args:
            observation: Single observation with shape [input_size] or 
                         batch of observations with shape [batch_size, input_size]
            alive_mask: Optional tensor indicating which agents are alive [pop_size]
                      
        Returns:
            Tensor of probabilities for each action with shape [pop_size, 1] or [batch_size, pop_size, 1]
        """
        # Check energy level
        has_energy = self.energy >= self.energy_cost_predict
        
        # Apply alive mask if provided - only alive agents can have energy
        if alive_mask is not None:
            has_energy = has_energy & alive_mask
            
        # Convert observation to tensor if it's a numpy array
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        # Ensure the observation is on the right device
        observation = observation.to(self.device)
        
        # Handle single observation vs batch differently
        if observation.dim() == 1:  # Single observation
            # Apply energy cost to agents with sufficient energy
            if self.evolvable_energy:
                self.energy[has_energy] -= self.energy_cost_predict[has_energy]
            else:
                self.energy[has_energy] -= self.energy_cost_predict
            
            # Forward pass
            with torch.no_grad():
                # Initialize output with zeros for all agents
                output = torch.zeros((self.pop_size, 1), device=self.device)
                
                # Only get predictions for agents with energy
                if has_energy.any():
                    # Expand observation for all agents regardless of energy
                    obs_expanded = observation.expand(self.pop_size, -1)
                    
                    # Run prediction for all agents (the inactive ones won't matter)
                    predictions = self.model(obs_expanded)
                    
                    # Apply energy mask to predictions
                    output = torch.where(
                        has_energy.unsqueeze(1),
                        torch.sigmoid(predictions),
                        output
                    )
                
                return output
        
        elif observation.dim() == 2:  # Batch of observations
            batch_size = observation.size(0)
            
            # Apply energy cost to agents with sufficient energy
            if self.evolvable_energy:
                self.energy[has_energy] -= self.energy_cost_predict[has_energy]
            else:
                self.energy[has_energy] -= self.energy_cost_predict
            
            # Forward pass
            with torch.no_grad():
                # Initialize output tensor with zeros
                output = torch.zeros((batch_size, self.pop_size, 1), device=self.device)
                
                # Only get predictions for agents with energy
                if has_energy.any():
                    # Create energy mask for batched processing
                    energy_mask = has_energy.unsqueeze(0).expand(batch_size, -1)
                    
                    # Expand observation for all agents regardless of energy
                    # [batch_size, input_size] -> [batch_size, pop_size, input_size]
                    obs_expanded = observation.unsqueeze(1).expand(-1, self.pop_size, -1)
                    
                    # Forward pass through model
                    predictions = self.model(obs_expanded)
                    
                    # Apply energy mask to output
                    output = torch.where(
                        energy_mask.unsqueeze(-1),
                        torch.sigmoid(predictions),
                        output
                    )
                
                return output
        else:
            raise ValueError(f"Observation must be 1D or 2D, got {observation.dim()}D")
    
    def learn(self, observation: torch.Tensor, reward: torch.Tensor, alive_mask: Optional[torch.Tensor] = None) -> None:
        """
        Learn from observation and reward.
        
        Args:
            observation: Observation tensor with shape [input_size], [pop_size, input_size], 
                         or [batch_size, input_size]
            reward: Reward tensor with shape [1], [pop_size], or [batch_size]
            alive_mask: Optional tensor indicating which agents are alive [pop_size]
        """
        # Check energy level
        has_energy = self.energy >= self.energy_cost_learn
        
        # Apply alive mask if provided
        if alive_mask is not None:
            has_energy = has_energy & alive_mask
            
        # If no agent has enough energy, return early
        if not has_energy.any():
            return
        
        # Ensure tensors are on the correct device
        observation = observation.to(self.device)
        reward = reward.to(self.device)
        
        # Single observation/reward case
        if observation.dim() == 1:
            # Create a batch with the full population
            obs_expanded = observation.expand(self.pop_size, -1)
            reward_expanded = reward.expand(self.pop_size) if not isinstance(reward, float) else torch.full((self.pop_size,), reward, device=self.device)
            
            # Forward pass - only for agents with energy
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Get predictions for all agents
                predictions = self.model(obs_expanded)
                predictions = torch.sigmoid(predictions)
                
                # Make target tensor
                target = reward_expanded.unsqueeze(-1)
                
                # Calculate loss only for agents with energy
                # Create a mask to zero out loss for agents without energy
                mask = has_energy.unsqueeze(-1).to(dtype=predictions.dtype)
                
                # Element-wise loss
                error = predictions - target
                loss = (error * error) * mask  # Squared error with mask
                
                # Calculate prediction error (for energy cost scaling)
                prediction_error = torch.abs(error).squeeze(-1)
                
                # Apply energy cost proportional to prediction error (for agents with energy)
                if self.evolvable_energy:
                    # Scale energy cost by prediction error and energy efficiency
                    energy_cost = torch.zeros_like(self.energy)
                    energy_cost[has_energy] = self.energy_cost_learn[has_energy] * (0.5 + 0.5 * prediction_error[has_energy])
                    # Apply energy cost
                    self.energy -= energy_cost
                else:
                    # Scale energy cost by prediction error
                    energy_cost = self.energy_cost_learn * (0.5 + 0.5 * prediction_error.mean().item())
                    # Apply energy cost
                    self.energy[has_energy] -= energy_cost
                
                # Mean loss for agents with energy
                loss = loss.sum() / (mask.sum() + 1e-8)
                
                # Add regularization
                if self.l1_reg > 0:
                    loss += self.l1_reg * self.model.l1_loss()
                if self.l2_reg > 0:
                    loss += self.l2_reg * self.model.l2_loss()
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Increment step count for agents with energy
                self.step_count[has_energy] += 1
        
        # Batch of observations case
        elif observation.dim() == 2:
            batch_size = observation.size(0)
            
            # Forward pass (for agents with energy only)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Expand observations for all agents
                # [batch_size, input_size] -> [batch_size, pop_size, input_size]
                obs_expanded = observation.unsqueeze(1).expand(-1, self.pop_size, -1)
                
                # Expand reward for all agents
                # [batch_size] -> [batch_size, pop_size]
                if reward.dim() == 1 and reward.size(0) == batch_size:
                    reward_expanded = reward.unsqueeze(1).expand(-1, self.pop_size)
                else:
                    # Assume scalar reward
                    reward_expanded = torch.full((batch_size, self.pop_size), reward.item() if torch.is_tensor(reward) else reward, device=self.device)
                
                # Run model forward pass 
                predictions = self.model(obs_expanded)
                predictions = torch.sigmoid(predictions)
                
                # Make target tensor [batch_size, pop_size, 1]
                target = reward_expanded.unsqueeze(-1)
                
                # Create energy mask [batch_size, pop_size, 1]
                mask = has_energy.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1).to(dtype=predictions.dtype)
                
                # Element-wise loss with mask
                error = predictions - target
                loss = (error * error) * mask  # Squared error with mask
                
                # Calculate prediction error (for energy cost scaling)
                prediction_error = torch.abs(error).mean(dim=0).squeeze(-1)  # [pop_size]
                
                # Apply energy cost proportional to prediction error
                if self.evolvable_energy:
                    # Scale energy cost by prediction error and energy efficiency
                    energy_cost = torch.zeros_like(self.energy)
                    energy_cost[has_energy] = self.energy_cost_learn[has_energy] * (0.5 + 0.5 * prediction_error[has_energy])
                    # Apply energy cost
                    self.energy -= energy_cost
                else:
                    # Scale energy cost by prediction error
                    energy_cost = self.energy_cost_learn * (0.5 + 0.5 * prediction_error.mean().item())
                    # Apply energy cost (only for agents with energy)
                    self.energy[has_energy] -= energy_cost
                
                # Mean loss for agents with energy
                loss = loss.sum() / (mask.sum() + 1e-8)
                
                # Add regularization
                if self.l1_reg > 0:
                    loss += self.l1_reg * self.model.l1_loss()
                if self.l2_reg > 0:
                    loss += self.l2_reg * self.model.l2_loss()
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Increment step count for agents with energy
                self.step_count[has_energy] += 1
        
        else:
            raise ValueError(f"Observation must be 1D or 2D, got {observation.dim()}D")
    
    def update_energy(self, performance_factor=None) -> None:
        """
        Update energy levels for all agents.
        
        Args:
            performance_factor: Optional tensor of performance factors [pop_size],
                                values should be between 0 and 1, where higher is better
        """
        if self.evolvable_energy:
            # Base recovery from agent genes
            recovery = self.energy_recovery
            
            # Apply performance-based bonus if provided
            if performance_factor is not None:
                # Ensure performance_factor is on the correct device
                if performance_factor.device != self.device:
                    performance_factor = performance_factor.to(self.device)
                
                # Bonus ranges from 0% to 50% of base recovery
                recovery = recovery * (1.0 + 0.5 * performance_factor)
            
            # Add recovery energy
            self.energy = torch.clamp(self.energy + recovery, max=self.base_energy_init)
        else:
            # Fixed recovery for all agents
            base_recovery = self.energy_recovery
            
            # Apply performance-based bonus if provided
            if performance_factor is not None:
                # Ensure performance_factor is on the correct device
                if performance_factor.device != self.device:
                    performance_factor = performance_factor.to(self.device)
                
                # Bonus ranges from 0% to 50% of base recovery
                recovery = base_recovery * (1.0 + 0.5 * performance_factor)
            else:
                recovery = base_recovery
            
            # Add recovery energy
            self.energy = torch.clamp(self.energy + recovery, max=self.energy_init)
        
        # Reset warnings for agents that have recovered energy
        self.warned_low_energy = self.warned_low_energy & (self.energy < self.energy_cost_learn if not self.evolvable_energy 
                                                          else self.energy < self.energy_cost_learn)
    
    def get_energy(self) -> torch.Tensor:
        """
        Get current energy levels.
        
        Returns:
            Tensor of energy levels for all agents
        """
        return self.energy
    
    def get_layer_parameters(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the weights and biases for a specific layer.
        
        Args:
            layer_idx: Index of the layer to get parameters for
            
        Returns:
            Tuple of (weights, biases) tensors for the specified layer
        """
        if layer_idx >= len(self.model.layers):
            raise ValueError(f"Invalid layer index: {layer_idx}, model has {len(self.model.layers)} layers")
        
        layer = self.model.layers[layer_idx]
        return layer.weight.detach(), layer.bias.detach()
    
    def set_layer_parameters(self, layer_idx: int, weights: torch.Tensor, biases: torch.Tensor) -> None:
        """
        Set the weights and biases for a specific layer.
        
        Args:
            layer_idx: Index of the layer to set parameters for
            weights: Tensor of shape [pop_size, output_size, input_size]
            biases: Tensor of shape [pop_size, output_size]
        """
        if layer_idx >= len(self.model.layers):
            raise ValueError(f"Invalid layer index: {layer_idx}, model has {len(self.model.layers)} layers")
        
        layer = self.model.layers[layer_idx]
        with torch.no_grad():
            layer.weight.copy_(weights)
            layer.bias.copy_(biases)
            
    def get_output_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the weights and biases for the output layer.
        
        Returns:
            Tuple of (weights, biases) tensors for the output layer
        """
        return self.model.output_layer.weight.detach(), self.model.output_layer.bias.detach()
    
    def set_output_parameters(self, weights: torch.Tensor, biases: torch.Tensor) -> None:
        """
        Set the weights and biases for the output layer.
        
        Args:
            weights: Tensor of shape [pop_size, 1, hidden_size]
            biases: Tensor of shape [pop_size, 1]
        """
        with torch.no_grad():
            self.model.output_layer.weight.copy_(weights)
            self.model.output_layer.bias.copy_(biases)
    
    def reset_state(self) -> None:
        """Reset agent state variables for a new generation."""
        # Don't reset energy levels between generations
        # self.energy = torch.full((self.pop_size,), self.energy_init, device=self.device)
        
        self.step_count = torch.zeros(self.pop_size, device=self.device)
        self.warned_low_energy = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        self.total_reward = torch.zeros(self.pop_size, device=self.device)
    
    def update_reward(self, rewards: torch.Tensor) -> None:
        """
        Update total rewards for agents.
        
        Args:
            rewards: Tensor of rewards for each agent [pop_size]
        """
        if not hasattr(self, 'total_reward'):
            self.total_reward = torch.zeros(self.pop_size, device=self.device)
            
        # Add rewards to total
        self.total_reward += rewards
        
        # Increment step count
        self.step_count += 1
    
    def get_lci(self) -> torch.Tensor:
        """
        Get the LCI (Learning Capacity Index) for all agents.
        LCI is calculated as total reward per step.
        
        Returns:
            Tensor of LCI values for all agents
        """
        if not hasattr(self, 'total_reward'):
            return torch.zeros(self.pop_size, device=self.device)
            
        # Avoid division by zero
        steps = torch.clamp(self.step_count, min=1.0)
        return self.total_reward / steps
    
    def get_alive_mask(self) -> torch.Tensor:
        """
        Get mask indicating which agents are alive (have energy > 0).
        
        Returns:
            Boolean tensor with True for alive agents
        """
        return self.energy > 0
    
    def get_step_count(self) -> torch.Tensor:
        """
        Get step count for all agents.
        
        Returns:
            Tensor of step counts
        """
        return self.step_count
    
    def save_agent(self, path: str) -> None:
        """
        Save agent state.
        
        Args:
            path: Path to save the agent state
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        model_path = f"{path}_model.pt"
        self.model.save_population(model_path)
        
        # Save agent state
        state = {
            'energy': self.energy,
            'step_count': self.step_count,
            'energy_cost_predict': self.energy_cost_predict,
            'energy_cost_learn': self.energy_cost_learn,
            'energy_init': self.energy_init,
            'energy_recovery': self.energy_recovery,
            'learning_rate': self.learning_rate,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'pop_size': self.pop_size,
            'input_size': self.input_size
        }
        torch.save(state, f"{path}_state.pt")
    
    def load_agent(self, path: str) -> None:
        """
        Load agent state.
        
        Args:
            path: Path to load the agent state from
        """
        # Load model
        model_path = f"{path}_model.pt"
        self.model.load_population(model_path)
        
        # Load agent state
        state = torch.load(f"{path}_state.pt", map_location=self.device)
        self.energy = state['energy'].to(self.device)
        self.step_count = state['step_count'].to(self.device)
        self.energy_cost_predict = state['energy_cost_predict']
        self.energy_cost_learn = state['energy_cost_learn']
        self.energy_init = state['energy_init']
        self.energy_recovery = state['energy_recovery']
        self.learning_rate = state['learning_rate']
        self.l1_reg = state['l1_reg']
        self.l2_reg = state['l2_reg']
        self.pop_size = state['pop_size']
        self.input_size = state['input_size']
        
        # Ensure optimizer is updated with new model parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Reset warnings
        self.warned_low_energy = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device) 