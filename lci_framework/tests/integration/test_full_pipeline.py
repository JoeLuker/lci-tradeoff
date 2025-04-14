"""
Integration test for the LCI framework

This test verifies that the main components of the system work together correctly.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np

from lci_framework.environments.markov_environment import MarkovEnvironment
from lci_framework.agents.lci_agent import LCIAgent
from lci_framework.core.evolution import LCIEvolution
from lci_framework.utils.config import create_default_config, setup_logging


class TestFullPipeline(unittest.TestCase):
    """Integration tests for the full LCI pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Setup logging
        setup_logging(log_file=os.path.join(self.test_dir, "test.log"))
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_agent_environment_interaction(self):
        """Test agent-environment interaction"""
        # Create environment
        env = MarkovEnvironment(n_states=10, seed=42)
        
        # Create agent
        agent = LCIAgent(
            input_size=env.n_states,
            learning_rate=0.01,
            hidden_size=16,
            n_layers=2,
            seed=42
        )
        
        # Reset environment
        state = env.reset()
        obs = env.get_observation(state)
        
        # Run interaction for a few steps
        total_reward = 0
        error_history = []
        
        for _ in range(10):
            # Agent predicts next state
            pred_state = agent.predict(obs)
            
            # Environment takes a step
            next_state, reward = env.step()
            next_obs = env.get_observation(next_state)
            
            # Agent learns
            error = agent.learn(obs, next_obs)
            error_history.append(error)
            
            # Agent receives reward
            agent.receive_reward(reward)
            total_reward += reward
            
            # Update observation
            obs = next_obs
        
        # Check that interaction produced reasonable results
        self.assertGreater(agent.energy, 0)  # Agent should still be alive
        self.assertGreater(len(error_history), 0)
        
        # Prediction errors should generally decrease as agent learns
        first_half = np.mean(error_history[:5])
        second_half = np.mean(error_history[5:])
        
        # This is a stochastic test and might occasionally fail
        # In general, the agent should improve over time
        self.assertLessEqual(second_half, first_half * 1.5)
        
    def test_evolution_pipeline(self):
        """Test the full evolution pipeline with a small configuration"""
        # Create a default configuration
        config = create_default_config()
        
        # Modify for faster testing
        config["environment"]["n_states"] = 5
        config["evolution"]["pop_size"] = 5
        config["evolution"]["n_generations"] = 2
        config["evolution"]["steps_per_eval"] = 10
        
        # Extract configuration
        env_config = config["environment"]
        evo_config = config["evolution"].copy()
        
        # Add output directory
        evo_config["output_dir"] = self.test_dir
        
        # Remove generation and step params which are passed separately
        n_generations = evo_config.pop("n_generations")
        steps_per_eval = evo_config.pop("steps_per_eval")
        
        # Create evolution
        evolution = LCIEvolution(env_config=env_config, **evo_config)
        
        # Run simulation
        evolution.run_simulation(n_generations=n_generations, steps_per_eval=steps_per_eval)
        
        # Verify results
        self.assertEqual(evolution.generation, n_generations)
        self.assertEqual(len(evolution.fitness_history), n_generations)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "stats_gen_0.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "lci_results_gen_0.png")))
        
    def test_agent_persistence(self):
        """Test saving and loading agents across environments"""
        # Create environment and agent
        env = MarkovEnvironment(n_states=5, seed=42)
        agent = LCIAgent(input_size=env.n_states, seed=42)
        
        # Train agent a bit
        state = env.reset()
        obs = env.get_observation(state)
        
        for _ in range(10):
            next_state, reward = env.step()
            next_obs = env.get_observation(next_state)
            agent.learn(obs, next_obs)
            obs = next_obs
        
        # Save agent
        model_path = os.path.join(self.test_dir, "test_agent.pt")
        agent.save_model(model_path)
        
        # Create a new observation
        test_obs = np.zeros(env.n_states)
        test_obs[2] = 1.0
        
        # Get prediction from original agent
        original_pred = agent.predict(test_obs)
        
        # Create a new agent and load the model
        new_agent = LCIAgent(input_size=env.n_states)
        new_agent.load_model(model_path)
        
        # Get prediction from new agent
        new_pred = new_agent.predict(test_obs)
        
        # Predictions should match
        self.assertEqual(original_pred, new_pred)
        
        # Create a new environment
        new_env = MarkovEnvironment(n_states=5, seed=43)  # Different seed
        
        # Check that loaded agent works in new environment
        state = new_env.reset()
        obs = new_env.get_observation(state)
        
        for _ in range(5):
            pred_state = new_agent.predict(obs)
            next_state, reward = new_env.step()
            next_obs = new_env.get_observation(next_state)
            new_agent.learn(obs, next_obs)
            obs = next_obs
            
        # Agent should still be functional
        self.assertTrue(new_agent.is_alive())


if __name__ == '__main__':
    unittest.main() 