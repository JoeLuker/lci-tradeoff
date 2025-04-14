"""
Unit tests for the LCIEvolution class
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path

from lci_framework.core.evolution import LCIEvolution
from lci_framework.agents.lci_agent import LCIAgent


class TestLCIEvolution(unittest.TestCase):
    """Test cases for the LCIEvolution class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Basic environment configuration
        self.env_config = {
            "n_states": 5,
            "sparse_transitions": True,
            "seed": 42
        }
        
        # Evolution parameters for testing (small values for faster tests)
        self.evolution = LCIEvolution(
            env_config=self.env_config,
            pop_size=5,
            mutation_rate=0.1,
            tournament_size=2,
            elitism=1,
            seed=42,
            output_dir=self.test_dir
        )
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test evolution initialization"""
        # Check that evolution is properly initialized
        self.assertEqual(self.evolution.pop_size, 5)
        self.assertEqual(self.evolution.mutation_rate, 0.1)
        self.assertEqual(self.evolution.tournament_size, 2)
        self.assertEqual(self.evolution.elitism, 1)
        
        # Check that population is created
        self.assertEqual(len(self.evolution.population), 5)
        
        # Check that all agents are of the correct type
        for agent in self.evolution.population:
            self.assertIsInstance(agent, LCIAgent)
            
        # Check that environment is created
        self.assertEqual(self.evolution.env.n_states, 5)
        
        # Check that histories are initialized
        self.assertEqual(self.evolution.generation, 0)
        self.assertEqual(len(self.evolution.fitness_history), 0)
        self.assertEqual(len(self.evolution.lci_history), 0)
        
    def test_evaluate_fitness(self):
        """Test fitness evaluation"""
        # Evaluate fitness
        results = self.evolution.evaluate_fitness(steps_per_eval=10)
        
        # Check results
        self.assertEqual(len(results), 5)  # One result per agent
        
        for result in results:
            # Check result structure
            self.assertIn("agent_id", result)
            self.assertIn("fitness", result)
            self.assertIn("reward", result)
            self.assertIn("steps_survived", result)
            self.assertIn("stability", result)
            self.assertIn("efficiency", result)
            self.assertIn("lci_balance", result)
            self.assertIn("lci_params", result)
            self.assertIn("final_energy", result)
            
        # Check that history is updated
        self.assertEqual(len(self.evolution.fitness_history), 1)
        self.assertEqual(len(self.evolution.lci_history), 1)
        
    def test_selection_and_reproduction(self):
        """Test selection and reproduction"""
        # First evaluate fitness
        results = self.evolution.evaluate_fitness(steps_per_eval=10)
        
        # Get initial population
        initial_pop = set(agent.agent_id for agent in self.evolution.population)
        
        # Perform selection and reproduction
        self.evolution.selection_and_reproduction(results)
        
        # Check that generation is incremented
        self.assertEqual(self.evolution.generation, 1)
        
        # Check that population size is maintained
        self.assertEqual(len(self.evolution.population), 5)
        
        # Check that some agents are new (reproduction occurred)
        new_pop = set(agent.agent_id for agent in self.evolution.population)
        self.assertNotEqual(initial_pop, new_pop)
        
        # With elitism=1, at least one agent should be preserved
        self.assertTrue(len(initial_pop.intersection(new_pop)) >= 1)
        
    def test_tournament_selection(self):
        """Test tournament selection"""
        # Create mock fitness results
        fitness_results = [
            {"agent_id": agent.agent_id, "fitness": i}
            for i, agent in enumerate(self.evolution.population)
        ]
        
        # Run tournament selection multiple times
        selected_ids = set()
        for _ in range(20):
            agent = self.evolution._tournament_selection(fitness_results)
            selected_ids.add(agent.agent_id)
            
        # With enough trials, most agents should be selected at least once
        # (stochastic test, might occasionally fail)
        self.assertGreater(len(selected_ids), 1)
        
    def test_crossover_and_mutate(self):
        """Test crossover and mutation"""
        # Get two parent agents
        parent1 = self.evolution.population[0]
        parent2 = self.evolution.population[1]
        
        # Create offspring
        offspring = self.evolution._crossover_and_mutate(parent1, parent2)
        
        # Check offspring type
        self.assertIsInstance(offspring, LCIAgent)
        
        # Check that offspring has different ID
        self.assertNotEqual(offspring.agent_id, parent1.agent_id)
        self.assertNotEqual(offspring.agent_id, parent2.agent_id)
        
        # Check that offspring has parameters within expected ranges
        self.assertGreater(offspring.learning_rate, 0)
        self.assertLess(offspring.learning_rate, 0.2)  # Upper bound with mutation
        
        self.assertGreater(offspring.hidden_size, 0)
        
        self.assertGreaterEqual(offspring.n_layers, 1)
        self.assertLessEqual(offspring.n_layers, 4)  # Upper bound with mutation
        
    def test_run_simulation(self):
        """Test running a short simulation"""
        # Run a very short simulation
        self.evolution.run_simulation(n_generations=2, steps_per_eval=5)
        
        # Check that results are updated
        self.assertEqual(self.evolution.generation, 2)
        self.assertEqual(len(self.evolution.fitness_history), 2)
        self.assertEqual(len(self.evolution.lci_history), 2)
        
        # Check that files are created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "stats_gen_0.json")))
        
    def test_plot_results(self):
        """Test plotting results"""
        # Run a short simulation to generate data
        self.evolution.run_simulation(n_generations=2, steps_per_eval=5)
        
        # Plot results
        plot_path = os.path.join(self.test_dir, "test_plot.png")
        self.evolution.plot_results(plot_path)
        
        # Check that plot file is created
        self.assertTrue(os.path.exists(plot_path))
        
    def test_save_results(self):
        """Test saving results"""
        # Run a short simulation to generate data
        self.evolution.run_simulation(n_generations=2, steps_per_eval=5)
        
        # Save results
        self.evolution.save_results("test")
        
        # Check that files are created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "stats_test.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "lci_results_test.png")))
        
    def test_input_validation(self):
        """Test input validation during initialization"""
        # Test invalid population size
        with self.assertRaises(ValueError):
            LCIEvolution(env_config=self.env_config, pop_size=0)
            
        # Test invalid mutation rate
        with self.assertRaises(ValueError):
            LCIEvolution(env_config=self.env_config, mutation_rate=2.0)
            
        # Test invalid tournament size
        with self.assertRaises(ValueError):
            LCIEvolution(env_config=self.env_config, pop_size=10, tournament_size=15)
            
        # Test invalid elitism
        with self.assertRaises(ValueError):
            LCIEvolution(env_config=self.env_config, pop_size=10, elitism=6)


if __name__ == '__main__':
    unittest.main() 