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
        
        # Set up device (MPS or CUDA only)
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("No GPU available. This implementation requires GPU acceleration.")
        
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
            x: Input tensor with shape [pop_size, in_features] or [batch_size, pop_size, in_features]
            
        Returns:
            Output tensor with shape [pop_size, out_features] or [batch_size, pop_size, out_features]
        """
        input_dim = x.dim()
        
        # Handle different input shapes
        if input_dim == 2:  # [pop_size, in_features]
            # Reshape x to [pop_size, 1, in_features] for batched matrix multiplication
            x_reshaped = x.unsqueeze(1)
            
            # Batched matrix multiplication: [pop_size, 1, in_features] @ [pop_size, out_features, in_features].transpose(-1, -2)
            # Result: [pop_size, 1, out_features]
            output = torch.bmm(x_reshaped, self.weight.transpose(-1, -2))
            
            # Reshape to [pop_size, out_features]
            output = output.squeeze(1)
            
            # Add bias if present
            if self.bias is not None:
                output += self.bias
        
        elif input_dim == 3:  # [batch_size, pop_size, in_features]
            batch_size = x.size(0)
            
            # Reshape for batched processing
            # [batch_size, pop_size, in_features] -> [batch_size * pop_size, in_features]
            x_flat = x.reshape(-1, x.size(-1))
            
            # Replicate weights for each batch element
            # [pop_size, out_features, in_features] -> [batch_size, pop_size, out_features, in_features]
            weight_expanded = self.weight.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # Reshape weights: [batch_size * pop_size, out_features, in_features]
            weight_flat = weight_expanded.reshape(-1, self.out_features, self.in_features)
            
            # Process each batch element and agent as a separate batch
            # [batch_size * pop_size, 1, in_features] @ [batch_size * pop_size, out_features, in_features].transpose(-1, -2)
            # Result: [batch_size * pop_size, 1, out_features]
            output = torch.bmm(x_flat.unsqueeze(1), weight_flat.transpose(-1, -2))
            
            # Reshape back to [batch_size, pop_size, out_features]
            output = output.squeeze(1).reshape(batch_size, self.pop_size, self.out_features)
            
            # Add bias if present
            if self.bias is not None:
                # Expand bias: [pop_size, out_features] -> [batch_size, pop_size, out_features]
                bias_expanded = self.bias.unsqueeze(0).expand(batch_size, -1, -1)
                output += bias_expanded
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got {input_dim}D")
        
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
        self.energy_cost_predict = energy_cost_predict
        self.energy_cost_learn = energy_cost_learn
        self.energy_init = energy_init
        self.energy_recovery = energy_recovery
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
        
    def predict(self, observation: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Make a prediction based on observation.
        
        Args:
            observation: Single observation with shape [input_size] or 
                         batch of observations with shape [batch_size, input_size]
                         
        Returns:
            Tensor of probabilities for each action with shape [pop_size, 1] or [batch_size, pop_size, 1]
        """
        # Check energy level
        has_energy = self.energy >= self.energy_cost_predict
        
        # Convert observation to tensor if it's a numpy array
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        # Ensure the observation is on the right device
        if observation.device != self.device:
            observation = observation.to(self.device)
        
        # Handle single observation vs batch
        if observation.dim() == 1:  # Single observation
            # Expand observation for each agent in the population
            observation_expanded = observation.unsqueeze(0).expand(self.pop_size, -1)
            
            # Only apply energy cost to agents with sufficient energy
            self.energy[has_energy] -= self.energy_cost_predict
            
            # Forward pass
            with torch.no_grad():
                # Only get predictions for agents with energy
                output = torch.zeros((self.pop_size, 1), device=self.device)
                if has_energy.any():
                    # Prepare observations for agents with energy
                    obs_with_energy = observation_expanded[has_energy]
                    
                    # Get predictions
                    predictions = self.model(obs_with_energy)
                    
                    # Store predictions
                    output[has_energy] = predictions
            
            # Return sigmoid of output
            return torch.sigmoid(output)
            
        elif observation.dim() == 2:  # Batch of observations
            batch_size = observation.size(0)
            
            # Expand observation for each agent in the population
            # [batch_size, input_size] -> [batch_size, pop_size, input_size]
            observation_expanded = observation.unsqueeze(1).expand(-1, self.pop_size, -1)
            
            # Apply energy cost to agents with sufficient energy
            # Handle this differently for batch mode - apply to all batch items for simplicity
            self.energy[has_energy] -= self.energy_cost_predict
            
            # Forward pass
            with torch.no_grad():
                # Only get predictions for agents with energy
                output = torch.zeros((batch_size, self.pop_size, 1), device=self.device)
                if has_energy.any():
                    # Create mask for agents with energy
                    energy_mask = has_energy.unsqueeze(0).expand(batch_size, -1)
                    
                    # Get observations for agents with energy
                    # This is a bit complex because we need to handle the batch dimension
                    # We'll flatten and then reshape
                    obs_flat = observation_expanded.reshape(batch_size * self.pop_size, -1)
                    energy_mask_flat = energy_mask.reshape(batch_size * self.pop_size)
                    obs_with_energy = obs_flat[energy_mask_flat]
                    
                    # Count how many agents have energy
                    agents_with_energy = has_energy.sum().item()
                    
                    # Reshape to [batch_size, agents_with_energy, input_size]
                    obs_with_energy = obs_with_energy.reshape(batch_size, agents_with_energy, -1)
                    
                    # Get predictions
                    predictions = self.model(obs_with_energy)
                    
                    # Store predictions
                    # This is also complex because we need to map back to the right indices
                    flat_output = output.reshape(batch_size * self.pop_size, 1)
                    flat_output[energy_mask_flat] = predictions.reshape(-1, 1)
                    output = flat_output.reshape(batch_size, self.pop_size, 1)
            
            # Return sigmoid of output
            return torch.sigmoid(output)
        
        else:
            raise ValueError(f"Observation must be 1D or 2D, got {observation.dim()}D")
    
    def learn(self, observation: torch.Tensor, reward: torch.Tensor) -> None:
        """
        Learn from observation and reward.
        
        Args:
            observation: Observation tensor with shape [input_size] or [batch_size, input_size]
            reward: Reward tensor with shape [1] or [batch_size]
        """
        # Check energy level
        has_energy = self.energy >= self.energy_cost_learn
        
        # Log warning for agents with insufficient energy
        if not has_energy.all() and not self.warned_low_energy.all():
            no_energy_count = (~has_energy & ~self.warned_low_energy).sum().item()
            if no_energy_count > 0:
                logger.warning(f"{no_energy_count} agents have insufficient energy for learning")
                # Update warned flag
                self.warned_low_energy = self.warned_low_energy | ~has_energy
        
        # If no agent has enough energy, return early
        if not has_energy.any():
            return
        
        # Ensure tensors are on the correct device
        if observation.device != self.device:
            observation = observation.to(self.device)
        if reward.device != self.device:
            reward = reward.to(self.device)
        
        # Apply energy cost
        self.energy[has_energy] -= self.energy_cost_learn
        
        # Handle single observation vs batch
        if observation.dim() == 1:  # Single observation
            # Expand observation for each agent in the population
            observation_expanded = observation.unsqueeze(0).expand(self.pop_size, -1)
            
            # Only learn for agents with energy
            if has_energy.any():
                # Prepare observations for agents with energy
                obs_with_energy = observation_expanded[has_energy]
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(obs_with_energy)
                predictions = torch.sigmoid(predictions)
                
                # Calculate loss
                # Use reward as target
                target = torch.full_like(predictions, reward.item())
                loss = F.mse_loss(predictions, target)
                
                # Add regularization
                if self.l1_reg > 0:
                    loss += self.l1_reg * self.model.l1_loss()
                if self.l2_reg > 0:
                    loss += self.l2_reg * self.model.l2_loss()
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Increment step count
                self.step_count[has_energy] += 1
        
        elif observation.dim() == 2:  # Batch of observations
            batch_size = observation.size(0)
            
            # Expand observation for each agent in the population
            # [batch_size, input_size] -> [batch_size, pop_size, input_size]
            observation_expanded = observation.unsqueeze(1).expand(-1, self.pop_size, -1)
            
            # Only learn for agents with energy
            if has_energy.any():
                # Create mask for agents with energy
                energy_mask = has_energy.unsqueeze(0).expand(batch_size, -1)
                
                # Get observations for agents with energy
                # We'll flatten and then reshape
                obs_flat = observation_expanded.reshape(batch_size * self.pop_size, -1)
                energy_mask_flat = energy_mask.reshape(batch_size * self.pop_size)
                obs_with_energy = obs_flat[energy_mask_flat]
                
                # Count how many agents have energy
                agents_with_energy = has_energy.sum().item()
                
                # Reshape to [batch_size, agents_with_energy, input_size]
                obs_with_energy = obs_with_energy.reshape(batch_size, agents_with_energy, -1)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(obs_with_energy)
                predictions = torch.sigmoid(predictions)
                
                # Calculate loss
                # Use reward as target, expand it for each agent
                target = reward.unsqueeze(1).expand(-1, agents_with_energy).unsqueeze(-1)
                loss = F.mse_loss(predictions, target)
                
                # Add regularization
                if self.l1_reg > 0:
                    loss += self.l1_reg * self.model.l1_loss()
                if self.l2_reg > 0:
                    loss += self.l2_reg * self.model.l2_loss()
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Increment step count
                self.step_count[has_energy] += 1
        
        else:
            raise ValueError(f"Observation must be 1D or 2D, got {observation.dim()}D")
    
    def update_energy(self) -> None:
        """
        Update energy levels for all agents.
        """
        # Add recovery energy
        self.energy = torch.clamp(self.energy + self.energy_recovery, max=self.energy_init)
        
        # Reset warnings for agents that have recovered energy
        self.warned_low_energy = self.warned_low_energy & (self.energy < self.energy_cost_learn)
    
    def get_energy(self) -> torch.Tensor:
        """
        Get current energy levels.
        
        Returns:
            Tensor of energy levels for all agents
        """
        return self.energy
    
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