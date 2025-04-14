"""
Unit tests for the MarkovEnvironment class
"""

import unittest
import numpy as np
from lci_framework.environments.markov_environment import MarkovEnvironment


class TestMarkovEnvironment(unittest.TestCase):
    """Test cases for the MarkovEnvironment class"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = MarkovEnvironment(n_states=10, sparse_transitions=True, seed=42)
        
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.n_states, 10)
        self.assertTrue(self.env.sparse_transitions)
        self.assertEqual(self.env.current_state, 0)
        self.assertEqual(len(self.env.volatility_history), 0)
        
    def test_reset(self):
        """Test environment reset"""
        # First step to a new state
        self.env.step()
        current_state = self.env.current_state
        
        # Reset should change the state
        new_state = self.env.reset()
        self.assertEqual(new_state, self.env.current_state)
        
        # Test reset to a specific state
        specific_state = 5
        new_state = self.env.reset(specific_state)
        self.assertEqual(new_state, specific_state)
        self.assertEqual(self.env.current_state, specific_state)
        
    def test_step(self):
        """Test environment step function"""
        # Reset to known state
        self.env.reset(0)
        
        # Step should return a next state and reward
        next_state, reward = self.env.step()
        
        # Verify the state has changed
        self.assertEqual(next_state, self.env.current_state)
        
        # Verify reward calculation (reward = state / n_states)
        self.assertAlmostEqual(reward, next_state / self.env.n_states)
        
    def test_get_observation(self):
        """Test get_observation function"""
        # Reset to known state
        state = 3
        self.env.reset(state)
        
        # Get observation
        obs = self.env.get_observation()
        
        # Verify one-hot encoding
        self.assertEqual(len(obs), self.env.n_states)
        self.assertEqual(np.sum(obs), 1.0)
        self.assertEqual(obs[state], 1.0)
        
        # Test with explicit state
        other_state = 7
        obs = self.env.get_observation(other_state)
        self.assertEqual(obs[other_state], 1.0)
        
    def test_update_environment(self):
        """Test environment update with volatility"""
        # Reset environment
        self.env.reset()
        
        # Update environment a few times
        for i in range(10):
            self.env.update_environment(i)
            
        # Check that volatility history is maintained
        self.assertEqual(len(self.env.volatility_history), 10)
        
    def test_volatility_phases(self):
        """Test volatility phase transitions"""
        # Set a simple volatility schedule for testing
        self.env.volatility_schedule = [
            (5, 0.0),  # 5 timesteps of stability
            (5, 0.5)   # 5 timesteps of medium volatility
        ]
        
        # Reset and run through the phases
        self.env.reset()
        
        # First phase (stability)
        for i in range(5):
            self.env.update_environment(i)
            self.assertEqual(self.env.current_volatility, 0.0)
            
        # Second phase (medium volatility)
        for i in range(5, 10):
            self.env.update_environment(i)
            self.assertEqual(self.env.current_volatility, 0.5)
            
        # Back to first phase (cycling)
        self.env.update_environment(10)
        self.assertEqual(self.env.current_volatility, 0.0)
        
    def test_set_volatility_schedule(self):
        """Test setting a custom volatility schedule"""
        new_schedule = [
            (10, 0.1),
            (10, 0.9)
        ]
        
        self.env.set_volatility_schedule(new_schedule)
        self.assertEqual(self.env.volatility_schedule, new_schedule)
        self.assertEqual(self.env.current_phase, 0)
        self.assertEqual(self.env.current_volatility, 0.1)
        
    def test_sparse_vs_dense(self):
        """Test both sparse and dense transition representations"""
        # Create environments with different representations
        sparse_env = MarkovEnvironment(n_states=5, sparse_transitions=True, seed=42)
        dense_env = MarkovEnvironment(n_states=5, sparse_transitions=False, seed=42)
        
        # Reset both to same state
        sparse_env.reset(0)
        dense_env.reset(0)
        
        # Run multiple steps and verify both implementations work
        for _ in range(10):
            s_state, s_reward = sparse_env.step()
            d_state, d_reward = dense_env.step()
            
            # States may differ due to different implementations
            self.assertIsInstance(s_state, (int, np.integer))
            self.assertIsInstance(d_state, (int, np.integer))
            
            # Rewards should follow the same formula
            self.assertAlmostEqual(s_reward, s_state / sparse_env.n_states)
            self.assertAlmostEqual(d_reward, d_state / dense_env.n_states)


if __name__ == '__main__':
    unittest.main() 