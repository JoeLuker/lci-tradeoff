import mlx.core as mx
import mlx.nn as nn
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
                 bias: bool = True):
        super().__init__()
        
        # Initialize weights for the entire population
        # Shape: [pop_size, out_features, in_features]
        self.weight = mx.random.normal(
            (pop_size, out_features, in_features)
        ) * 0.02
        
        if bias:
            # Shape: [pop_size, out_features]
            self.bias = mx.zeros((pop_size, out_features))
        else:
            self.bias = None
        
        self.in_features = in_features
        self.out_features = out_features
        self.pop_size = pop_size
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the population linear layer.
        
        Args:
            x: Input tensor with shape [pop_size, batch_size, in_features]
            
        Returns:
            Output tensor with shape [pop_size, batch_size, out_features]
        """
        # Debug input
        logger.debug(f"PopulationLinear input: type={type(x)}, shape={getattr(x, 'shape', None)}")
        
        # Handle different input shapes
        if x.ndim == 2:  # [pop_size, in_features] - single sample case
            logger.debug("PopulationLinear 2D input case")
            
            # Reshape for matmul: [pop_size, 1, in_features]
            x_reshaped = mx.expand_dims(x, 1)
            logger.debug(f"After expand_dims: type={type(x_reshaped)}, shape={getattr(x_reshaped, 'shape', None)}")
            
            # Get transposed weights
            transposed_weight = mx.transpose(self.weight, (0, 2, 1))
            logger.debug(f"Transposed weight: type={type(transposed_weight)}, shape={getattr(transposed_weight, 'shape', None)}")
            
            # Batched matrix multiplication
            output = mx.matmul(x_reshaped, transposed_weight)
            logger.debug(f"After matmul: type={type(output)}, shape={getattr(output, 'shape', None)}")
            
            # Reshape to remove singleton dimension: [pop_size, out_features]
            output = mx.squeeze(output, 1)
            logger.debug(f"After squeeze: type={type(output)}, shape={getattr(output, 'shape', None)}")
            
            # Add bias
            if self.bias is not None:
                output = output + self.bias
                logger.debug(f"After bias: type={type(output)}, shape={getattr(output, 'shape', None)}")
                
        elif x.ndim == 3:  # [pop_size, batch_size, in_features] - batch case
            logger.debug("PopulationLinear 3D input case")
            
            # Get transposed weights
            transposed_weight = mx.transpose(self.weight, (0, 2, 1))
            logger.debug(f"Transposed weight: type={type(transposed_weight)}, shape={getattr(transposed_weight, 'shape', None)}")
            
            # Batched matrix multiplication
            output = mx.matmul(x, transposed_weight)
            logger.debug(f"After matmul: type={type(output)}, shape={getattr(output, 'shape', None)}")
            
            # Add bias
            if self.bias is not None:
                bias_expanded = mx.expand_dims(self.bias, 1)
                logger.debug(f"Expanded bias: type={type(bias_expanded)}, shape={getattr(bias_expanded, 'shape', None)}")
                output = output + bias_expanded
                logger.debug(f"After bias: type={type(output)}, shape={getattr(output, 'shape', None)}")
                
        else:
            raise ValueError(f"Expected input tensor with 2 or 3 dimensions, got {x.ndim}")
        
        # Final check
        if not isinstance(output, mx.array):
            logger.error(f"PopulationLinear returning non-array type: {type(output)}")
            if isinstance(output, tuple):
                logger.warning("Converting tuple to array")
                # Force conversion of tuple to array
                output = mx.array(output[0] if len(output) > 0 else 0.0)
        
        logger.debug(f"PopulationLinear output: type={type(output)}, shape={getattr(output, 'shape', None)}")
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
                 dropout_rate: float = 0.1):
        """
        Initialize a population of neural networks.
        
        Args:
            pop_size: Number of neural networks in the population
            input_size: Size of the input features
            hidden_size: Size of the hidden layers
            n_layers: Number of hidden layers
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # Create layers
        self.layers = [
            PopulationLinear(
                pop_size=pop_size, 
                in_features=input_size if i == 0 else hidden_size, 
                out_features=hidden_size
            )
            for i in range(n_layers)
        ]
        
        # Output layer
        self.output_layer = PopulationLinear(
            pop_size=pop_size, 
            in_features=hidden_size, 
            out_features=1  # Single output per network
        )
    
    def __call__(self, x: mx.array) -> mx.array:
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
        # Forward pass through hidden layers
        for layer in self.layers:
            x = nn.relu(layer(x))
            # Apply dropout if in training mode
            if self.dropout_rate > 0:
                mask = mx.random.uniform(shape=x.shape) > self.dropout_rate
                x = x * mask * (1.0 / (1.0 - self.dropout_rate))
        
        # Forward pass through output layer
        output = self.output_layer(x)
        
        return output
    
    def l1_loss(self) -> mx.array:
        """
        Calculate L1 regularization loss for all parameters.
        
        Returns:
            L1 regularization loss
        """
        l1_reg = 0.0
        
        # Sum L1 norms of all weights
        for layer in self.layers:
            l1_reg += mx.sum(mx.abs(layer.weight))
            if layer.bias is not None:
                l1_reg += mx.sum(mx.abs(layer.bias))
        
        # Add output layer parameters
        l1_reg += mx.sum(mx.abs(self.output_layer.weight))
        if self.output_layer.bias is not None:
            l1_reg += mx.sum(mx.abs(self.output_layer.bias))
        
        return l1_reg
    
    def l2_loss(self) -> mx.array:
        """
        Calculate L2 regularization loss for all parameters.
        
        Returns:
            L2 regularization loss
        """
        l2_reg = 0.0
        
        # Sum squared norms of all weights
        for layer in self.layers:
            l2_reg += mx.sum(mx.square(layer.weight))
            if layer.bias is not None:
                l2_reg += mx.sum(mx.square(layer.bias))
        
        # Add output layer parameters
        l2_reg += mx.sum(mx.square(self.output_layer.weight))
        if self.output_layer.bias is not None:
            l2_reg += mx.sum(mx.square(self.output_layer.bias))
        
        return l2_reg
    
    def save_population(self, path: str) -> None:
        """
        Save the population parameters to disk.
        
        Args:
            path: Path to save the parameters
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # For each parameter, save it separately
        for i, layer in enumerate(self.layers):
            mx.save(f"{path}_layer_{i}_weight", layer.weight)
            if layer.bias is not None:
                mx.save(f"{path}_layer_{i}_bias", layer.bias)
        
        # Save output layer parameters
        mx.save(f"{path}_output_weight", self.output_layer.weight)
        if self.output_layer.bias is not None:
            mx.save(f"{path}_output_bias", self.output_layer.bias)
    
    def load_population(self, path: str) -> None:
        """
        Load the population parameters from disk.
        
        Args:
            path: Path to load parameters from
        """
        # Load hidden layer parameters
        for i, layer in enumerate(self.layers):
            layer.weight = mx.load(f"{path}_layer_{i}_weight")
            if layer.bias is not None:
                layer.bias = mx.load(f"{path}_layer_{i}_bias")
        
        # Load output layer parameters
        self.output_layer.weight = mx.load(f"{path}_output_weight")
        if self.output_layer.bias is not None:
            self.output_layer.bias = mx.load(f"{path}_output_bias")

class TensorLCIAgent:
    """
    A vectorized implementation of the LCI agent using MLX arrays.
    This implementation can process an entire population of agents in parallel.
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
                 evolvable_energy: bool = True):
        """
        Initialize a population of LCI agents.
        
        Args:
            pop_size: Number of agents in the population
            input_size: Size of the input (state) vector
            energy_cost_predict: Energy cost for making a prediction
            energy_cost_learn: Energy cost for learning
            energy_init: Initial energy level
            energy_recovery: Energy recovery rate per step
            learning_rate: Learning rate for neural network updates
            hidden_size: Size of hidden layers in neural networks
            n_layers: Number of hidden layers in neural networks
            dropout_rate: Dropout probability
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            learn_prob: Learning probability
            evolvable_energy: Whether energy parameters can evolve
        """
        self.pop_size = pop_size
        self.input_size = input_size
        
        # Create neural network population
        self.policy_net = NeuralPopulation(
            pop_size=pop_size,
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )
        
        # Initialize energy parameters
        if evolvable_energy:
            self.energy_cost_predict = mx.ones((pop_size, 1)) * energy_cost_predict
            self.energy_cost_learn = mx.ones((pop_size, 1)) * energy_cost_learn
            self.energy_recovery = mx.ones((pop_size, 1)) * energy_recovery
        else:
            self.energy_cost_predict = energy_cost_predict
            self.energy_cost_learn = energy_cost_learn
            self.energy_recovery = energy_recovery
        
        self.energy_init = energy_init
        self.evolvable_energy = evolvable_energy
        
        # Current energy levels
        self.energy = mx.ones((pop_size,)) * energy_init
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learn_prob = learn_prob
        
        # Tracking variables
        self.cumulative_reward = mx.zeros((pop_size,))
        self.step_count = mx.zeros((pop_size,), dtype=mx.int32)
        self.lci_values = mx.zeros((pop_size,))
        self.learn_decisions = mx.zeros((pop_size,), dtype=mx.int32)
        
        # Create mask to track alive agents
        self.alive_mask = mx.ones((pop_size,), dtype=mx.bool_)
        
        # Training step counter
        self.train_step = 0
    
    def predict(self, observation: mx.array, alive_mask: Optional[mx.array] = None) -> mx.array:
        """
        Make a prediction for all agents in the population.
        
        Args:
            observation: Input observation with shape [pop_size, input_size]
                         or [batch_size, pop_size, input_size]
            alive_mask: Mask of alive agents, shape [pop_size] or None
                       If None, uses the internal alive_mask
                
        Returns:
            Action values with shape [pop_size, 1] or [batch_size, pop_size, 1]
        """
        # Use the internal alive mask if not provided
        if alive_mask is None:
            alive_mask = self.alive_mask
        
        # Reduce energy for prediction (for all alive agents)
        if isinstance(self.energy_cost_predict, mx.array):
            # Batch energy cost reduction for evolved parameters
            energy_cost = mx.squeeze(self.energy_cost_predict, axis=1)
            # Only reduce energy for alive agents
            energy_reduction = energy_cost * alive_mask
            self.energy = mx.maximum(0, self.energy - energy_reduction)
        else:
            # Scalar energy cost
            self.energy = mx.maximum(0, self.energy - (self.energy_cost_predict * alive_mask))
        
        # Get policy predictions
        action_values = self.policy_net(observation)
        
        # Update step count for alive agents
        self.step_count = self.step_count + mx.array(alive_mask, dtype=mx.int32)
        
        return action_values
    
    def learn(self, observation: mx.array, reward: mx.array, alive_mask: Optional[mx.array] = None) -> None:
        """
        Update the model based on the observed reward.
        
        Args:
            observation: Input observation with shape [pop_size, input_size] 
            reward: Reward for each agent with shape [pop_size]
            alive_mask: Boolean mask indicating which agents are alive, shape [pop_size]
        
        Returns:
            None
        """
        try:
            # Use the internal alive mask if not provided
            if alive_mask is None:
                alive_mask = self.alive_mask
                
            # Randomly determine which agents will learn this time
            # FIX: Safe broadcasting for learn_mask creation
            random_values = mx.random.uniform(shape=(self.pop_size,))
            learn_threshold = mx.ones(random_values.shape) * self.learn_prob
            learn_prob_check = random_values < learn_threshold
            learn_mask = mx.logical_and(learn_prob_check, alive_mask)
            
            # Only continue if there are agents that want to learn
            if mx.sum(learn_mask) == 0:
                return
                
            # Check if agents have enough energy to learn
            if isinstance(self.energy_cost_learn, mx.array):
                # Batch energy check for evolved parameters
                energy_cost = mx.squeeze(self.energy_cost_learn, axis=1)
                # FIX: Safe comparison for can_learn_mask
                energy_check = self.energy >= energy_cost
                can_learn_mask = mx.logical_and(energy_check, learn_mask)
            else:
                # Scalar energy check
                # FIX: Safe scalar comparison
                energy_threshold = mx.ones(self.energy.shape) * self.energy_cost_learn
                energy_check = self.energy >= energy_threshold
                can_learn_mask = mx.logical_and(energy_check, learn_mask)
                
            # Only continue if there are agents that can learn
            if mx.sum(can_learn_mask) == 0:
                return
                
            # Reduce energy for learning agents
            if isinstance(self.energy_cost_learn, mx.array):
                # Batch energy reduction for evolved parameters
                energy_cost = mx.squeeze(self.energy_cost_learn, axis=1)
                # FIX: Safe multiplication
                energy_reduction = mx.multiply(energy_cost, can_learn_mask.astype(mx.float32))
                self.energy = mx.maximum(mx.zeros(self.energy.shape), self.energy - energy_reduction)
            else:
                # Scalar energy reduction
                # FIX: Safe scalar reduction
                cost_amount = mx.ones(self.energy.shape) * self.energy_cost_learn
                energy_reduction = mx.multiply(cost_amount, can_learn_mask.astype(mx.float32))
                self.energy = mx.maximum(mx.zeros(self.energy.shape), self.energy - energy_reduction)
                
            # Update learn decision counter
            # FIX: Safe integer conversion
            learn_increments = can_learn_mask.astype(mx.int32)
            self.learn_decisions = self.learn_decisions + learn_increments
            
            # Define loss function using MLX's functional approach
            def loss_fn(params):
                # Extract parameters for policy network
                policy_params = {}
                idx = 0
                for i, layer in enumerate(self.policy_net.layers):
                    if hasattr(layer, 'weight'):
                        policy_params[f'layer_{i}_weight'] = params[idx]
                        idx += 1
                    if hasattr(layer, 'bias'):
                        policy_params[f'layer_{i}_bias'] = params[idx]
                        idx += 1
                        
                # Extract output layer parameters
                policy_params['output_weight'] = params[idx]
                idx += 1
                if hasattr(self.policy_net.output_layer, 'bias'):
                    policy_params['output_bias'] = params[idx]
                
                # Forward pass with these parameters
                # This is simplified and would need to be implemented properly
                # based on how your forward pass works with specific parameters
                pred = self.policy_net(observation)
                
                # Compute MSE loss only for learning agents
                squared_errors = mx.square(pred - reward.reshape(-1, 1))
                # FIX: Ensure safe operations for loss calculation
                mask_reshaped = mx.reshape(can_learn_mask, (-1, 1)).astype(mx.float32)
                masked_errors = mx.multiply(squared_errors, mask_reshaped)
                # Avoid division by zero
                divisor = mx.maximum(mx.sum(can_learn_mask), mx.array(1))
                mse_loss = mx.sum(masked_errors) / divisor
                
                # Add regularization
                l1_loss = self.policy_net.l1_loss() * self.l1_reg
                l2_loss = self.policy_net.l2_loss() * self.l2_reg
                
                return mse_loss + l1_loss + l2_loss
            
            # Get model parameters as a list
            params = []
            for layer in self.policy_net.layers:
                if hasattr(layer, 'weight'):
                    params.append(layer.weight)
                if hasattr(layer, 'bias'):
                    params.append(layer.bias)
                    
            # Add output layer parameters
            params.append(self.policy_net.output_layer.weight)
            if hasattr(self.policy_net.output_layer, 'bias'):
                params.append(self.policy_net.output_layer.bias)
                
            # Compute gradients
            grads = mx.grad(loss_fn)(params)
            
            # Update parameters with gradients
            for i, (param, grad) in enumerate(zip(params, grads)):
                params[i] = param - self.learning_rate * grad
                
            # Update model parameters
            idx = 0
            for layer in self.policy_net.layers:
                if hasattr(layer, 'weight'):
                    layer.weight = params[idx]
                    idx += 1
                if hasattr(layer, 'bias'):
                    layer.bias = params[idx]
                    idx += 1
                    
            # Update output layer parameters
            self.policy_net.output_layer.weight = params[idx]
            idx += 1
            if hasattr(self.policy_net.output_layer, 'bias'):
                self.policy_net.output_layer.bias = params[idx]
                
            # Increment training step counter
            self.train_step += 1
            
        except Exception as e:
            # Provide more detailed error information
            import traceback
            print(f"Error in learn method: {e}")
            traceback.print_exc()
            # Don't re-raise - let simulation continue despite learning failures
    
    def update_energy(self, performance_factor=None) -> None:
        """
        Update agent energy levels based on recovery rate.
        
        Args:
            performance_factor: Optional factor to scale energy recovery 
                               based on performance (not implemented yet)
        """
        # Apply energy recovery for alive agents
        if isinstance(self.energy_recovery, mx.array):
            # Batch energy recovery for evolved parameters
            recovery_rate = mx.squeeze(self.energy_recovery, axis=1)
            energy_recovery = recovery_rate * self.alive_mask
        else:
            # Scalar energy recovery
            energy_recovery = self.energy_recovery * self.alive_mask
            
        # Update energy levels
        self.energy = self.energy + energy_recovery
    
    def get_energy(self) -> mx.array:
        """
        Get current energy levels for all agents.
        
        Returns:
            Energy levels with shape [pop_size]
        """
        return self.energy
    
    def get_layer_parameters(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """
        Get parameters for a specific hidden layer.
        
        Args:
            layer_idx: Index of the hidden layer
            
        Returns:
            Tuple of (weights, biases) for the specified layer
        """
        if layer_idx >= len(self.policy_net.layers):
            raise ValueError(f"Layer index {layer_idx} out of range (max {len(self.policy_net.layers)-1})")
            
        layer = self.policy_net.layers[layer_idx]
        return layer.weight, layer.bias
    
    def set_layer_parameters(self, layer_idx: int, weights: mx.array, biases: mx.array) -> None:
        """
        Set parameters for a specific hidden layer.
        
        Args:
            layer_idx: Index of the hidden layer
            weights: Weights for the layer with shape [pop_size, out_features, in_features]
            biases: Biases for the layer with shape [pop_size, out_features]
        """
        if layer_idx >= len(self.policy_net.layers):
            raise ValueError(f"Layer index {layer_idx} out of range (max {len(self.policy_net.layers)-1})")
            
        layer = self.policy_net.layers[layer_idx]
        layer.weight = weights
        layer.bias = biases
    
    def get_output_parameters(self) -> Tuple[mx.array, mx.array]:
        """
        Get parameters for the output layer.
        
        Returns:
            Tuple of (weights, biases) for the output layer
        """
        return self.policy_net.output_layer.weight, self.policy_net.output_layer.bias
    
    def set_output_parameters(self, weights: mx.array, biases: mx.array) -> None:
        """
        Set parameters for the output layer.
        
        Args:
            weights: Weights for the output layer with shape [pop_size, out_features, in_features]
            biases: Biases for the output layer with shape [pop_size, out_features]
        """
        self.policy_net.output_layer.weight = weights
        self.policy_net.output_layer.bias = biases
    
    def reset_state(self) -> None:
        """
        Reset the agent state (not the learned parameters).
        """
        # Reset energy levels
        self.energy = mx.ones((self.pop_size,)) * self.energy_init
        
        # Reset tracking variables
        self.cumulative_reward = mx.zeros((self.pop_size,))
        self.step_count = mx.zeros((self.pop_size,), dtype=mx.int32)
        self.lci_values = mx.zeros((self.pop_size,))
        self.learn_decisions = mx.zeros((self.pop_size,), dtype=mx.int32)
        self.alive_mask = mx.ones((self.pop_size,), dtype=mx.bool_)
    
    def update_reward(self, rewards: mx.array) -> None:
        """
        Update cumulative rewards for each agent.
        
        Args:
            rewards: Rewards for each agent with shape [pop_size]
        """
        # Update rewards for all alive agents
        self.cumulative_reward = self.cumulative_reward + (rewards * self.alive_mask)
        
        # Update LCI values (reward per step)
        steps_safe = mx.maximum(self.step_count, 1)  # Avoid division by zero
        self.lci_values = self.cumulative_reward / steps_safe
    
    def get_lci(self) -> mx.array:
        """
        Get the current LCI values (Learning, Computation, Information tradeoff).
        
        These are computed as reward per step, measuring how efficiently
        the agent is balancing its learning and action selection.
        
        Returns:
            LCI values with shape [pop_size]
        """
        return self.lci_values
    
    def get_alive_mask(self) -> mx.array:
        """
        Get mask indicating which agents are alive.
        
        Returns:
            Boolean mask with shape [pop_size]
        """
        return self.alive_mask
    
    def get_step_count(self) -> mx.array:
        """
        Get the number of steps taken by each agent.
        
        Returns:
            Step count with shape [pop_size]
        """
        return self.step_count
    
    def save_agent(self, path: str) -> None:
        """
        Save the agent parameters to disk.
        
        Args:
            path: Path to save the parameters
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save policy network parameters
        self.policy_net.save_population(path)
        
        # Save other agent parameters individually
        mx.save(f"{path}_energy", self.energy)
        mx.save(f"{path}_cumulative_reward", self.cumulative_reward)
        mx.save(f"{path}_step_count", self.step_count)
        mx.save(f"{path}_lci_values", self.lci_values)
        mx.save(f"{path}_learn_decisions", self.learn_decisions)
        mx.save(f"{path}_alive_mask", self.alive_mask)
        
        # Save evolvable energy parameters if applicable
        if self.evolvable_energy:
            if isinstance(self.energy_cost_predict, mx.array):
                mx.save(f"{path}_energy_cost_predict", self.energy_cost_predict)
            if isinstance(self.energy_cost_learn, mx.array):
                mx.save(f"{path}_energy_cost_learn", self.energy_cost_learn)
            if isinstance(self.energy_recovery, mx.array):
                mx.save(f"{path}_energy_recovery", self.energy_recovery)
    
    def load_agent(self, path: str) -> None:
        """
        Load agent parameters from disk.
        
        Args:
            path: Path to load parameters from
        """
        # Load policy network parameters
        self.policy_net.load_population(path)
        
        # Load other agent parameters
        self.energy = mx.load(f"{path}_energy")
        self.cumulative_reward = mx.load(f"{path}_cumulative_reward")
        self.step_count = mx.load(f"{path}_step_count")
        self.lci_values = mx.load(f"{path}_lci_values")
        self.learn_decisions = mx.load(f"{path}_learn_decisions")
        self.alive_mask = mx.load(f"{path}_alive_mask")
        
        # Load evolvable energy parameters if applicable
        if self.evolvable_energy:
            try:
                self.energy_cost_predict = mx.load(f"{path}_energy_cost_predict")
                self.energy_cost_learn = mx.load(f"{path}_energy_cost_learn")
                self.energy_recovery = mx.load(f"{path}_energy_recovery")
            except:
                # If files don't exist, keep the current values
                pass