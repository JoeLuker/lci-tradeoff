"""
Unit tests for the LCIAgent class
"""

import unittest
import numpy as np
import torch
import os
import tempfile
from lci_framework.agents.lci_agent import LCIAgent, NeuralModel, BaseAgent


class TestNeuralModel(unittest.TestCase):
    """Test cases for the NeuralModel class"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = NeuralModel(input_size=10, hidden_size=16, n_layers=2, dropout_rate=0.1)
        
        # Check model structure
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model.network, torch.nn.Sequential)
        
        # Check forward pass
        x = torch.randn(10)
        output = model(x)
        self.assertEqual(output.shape, torch.Size([10]))


class TestLCIAgent(unittest.TestCase):
    """Test cases for the LCIAgent class"""
    
    def setUp(self):
        """Set up test environment"""
        self.input_size = 10
        self.agent = LCIAgent(
            input_size=self.input_size,
            learning_rate=0.01,
            hidden_size=16,
            n_layers=2,
            l1_reg=0.001,
            l2_reg=0.001,
            dropout_rate=0.1,
            initial_energy=100.0,
            agent_id=1,
            seed=42
        )
        
    def test_initialization(self):
        """Test agent initialization"""
        # Check that the agent is properly initialized
        self.assertEqual(self.agent.input_size, self.input_size)
        self.assertEqual(self.agent.learning_rate, 0.01)
        self.assertEqual(self.agent.hidden_size, 16)
        self.assertEqual(self.agent.n_layers, 2)
        self.assertEqual(self.agent.l1_reg, 0.001)
        self.assertEqual(self.agent.l2_reg, 0.001)
        self.assertEqual(self.agent.dropout_rate, 0.1)
        self.assertEqual(self.agent.energy, 100.0)
        self.assertEqual(self.agent.agent_id, 1)
        
        # Check LCI parameters
        self.assertIn("L", self.agent.lci_params)
        self.assertIn("C", self.agent.lci_params)
        self.assertIn("I", self.agent.lci_params)
        
        # Check model and optimizer
        self.assertIsInstance(self.agent.model, NeuralModel)
        self.assertIsInstance(self.agent.optimizer, torch.optim.Adam)
        
    def test_inheritance(self):
        """Test that LCIAgent inherits from BaseAgent"""
        self.assertIsInstance(self.agent, BaseAgent)
        
    def test_predict(self):
        """Test agent prediction"""
        # Create a one-hot observation
        obs = np.zeros(self.input_size)
        obs[2] = 1.0
        
        # Get prediction
        pred_state = self.agent.predict(obs)
        
        # Check prediction type
        self.assertIsInstance(pred_state, (int, np.integer))
        
        # Check energy consumption
        self.assertLess(self.agent.energy, 100.0)
        
    def test_learn(self):
        """Test agent learning"""
        # Create observations
        obs = np.zeros(self.input_size)
        obs[2] = 1.0
        
        next_obs = np.zeros(self.input_size)
        next_obs[5] = 1.0
        
        # Initial energy
        initial_energy = self.agent.energy
        
        # Learn
        loss = self.agent.learn(obs, next_obs)
        
        # Check loss
        self.assertGreaterEqual(loss, 0.0)
        
        # Check energy consumption
        self.assertLess(self.agent.energy, initial_energy)
        
    def test_receive_reward(self):
        """Test agent reward processing"""
        # Initial energy
        initial_energy = self.agent.energy
        
        # Receive reward
        self.agent.receive_reward(1.0)
        
        # Check energy increase
        self.assertGreater(self.agent.energy, initial_energy)
        
        # Check reward history
        self.assertEqual(len(self.agent.reward_history), 1)
        self.assertEqual(self.agent.reward_history[0], 1.0)
        
    def test_is_alive(self):
        """Test agent alive status"""
        # Initially alive
        self.assertTrue(self.agent.is_alive())
        
        # Set energy to negative
        self.agent.energy = -1.0
        
        # Should be dead
        self.assertFalse(self.agent.is_alive())
        
    def test_get_state(self):
        """Test getting agent state"""
        # Get state
        state = self.agent.get_state()
        
        # Check state structure
        self.assertIsInstance(state, dict)
        self.assertIn("id", state)
        self.assertIn("energy", state)
        self.assertIn("lci_params", state)
        self.assertIn("avg_reward", state)
        self.assertIn("avg_error", state)
        
    def test_get_lci_balance(self):
        """Test LCI balance calculation"""
        # Get balance
        balance = self.agent.get_lci_balance()
        
        # Check balance
        self.assertGreaterEqual(balance, 0.0)
        self.assertLessEqual(balance, 1.0)
        
    def test_save_load_model(self):
        """Test saving and loading model"""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
            
        try:
            # Create a test observation and get initial prediction
            obs = np.zeros(self.input_size)
            obs[3] = 1.0
            
            initial_pred = self.agent.predict(obs)
            
            # Save model
            self.agent.save_model(model_path)
            
            # Create new agent with same architecture parameters
            new_agent = LCIAgent(
                input_size=self.input_size,
                learning_rate=0.02,  # Different learning rate is fine
                hidden_size=self.agent.hidden_size,  # Must match original
                n_layers=self.agent.n_layers,        # Must match original
                dropout_rate=self.agent.dropout_rate # Must match original
            )
            
            # Load saved model
            new_agent.load_model(model_path)
            
            # Check that parameters were properly loaded
            self.assertEqual(new_agent.agent_id, self.agent.agent_id)
            
            # Check that loaded model produces same prediction
            new_pred = new_agent.predict(obs)
            self.assertEqual(new_pred, initial_pred)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
                
    def test_input_validation(self):
        """Test input validation during initialization"""
        # Test invalid input size
        with self.assertRaises(ValueError):
            LCIAgent(input_size=0)
            
        # Test invalid learning rate
        with self.assertRaises(ValueError):
            LCIAgent(input_size=10, learning_rate=0)
            
        # Test invalid hidden size
        with self.assertRaises(ValueError):
            LCIAgent(input_size=10, hidden_size=0)
            
        # Test invalid number of layers
        with self.assertRaises(ValueError):
            LCIAgent(input_size=10, n_layers=0)
            
        # Test invalid L1 regularization
        with self.assertRaises(ValueError):
            LCIAgent(input_size=10, l1_reg=2.0)
            
        # Test invalid L2 regularization
        with self.assertRaises(ValueError):
            LCIAgent(input_size=10, l2_reg=-0.1)
            
        # Test invalid dropout rate
        with self.assertRaises(ValueError):
            LCIAgent(input_size=10, dropout_rate=1.5)
            
    def test_energy_depletion(self):
        """Test agent behavior when energy is depleted"""
        # Deplete energy
        self.agent.energy = 0.0
        
        # Create test data
        obs = np.zeros(self.input_size)
        obs[2] = 1.0
        
        next_obs = np.zeros(self.input_size)
        next_obs[5] = 1.0
        
        # Prediction with no energy should return default value
        pred = self.agent.predict(obs)
        self.assertEqual(pred, 0)
        
        # Learning with no energy should return zero loss
        loss = self.agent.learn(obs, next_obs)
        self.assertEqual(loss, 0.0)


if __name__ == '__main__':
    unittest.main() 