"""
LCI Framework: GPU-accelerated Learning, Cognition, and Intelligence Framework

A tensor-based implementation leveraging GPU acceleration for evolutionary simulations
with neural networks and reinforcement learning.
"""

from lci_framework.core.evolution import TensorEvolution
from lci_framework.agents.agent import NeuralPopulation, TensorLCIAgent
from lci_framework.environments.environment import VectorizedMarkovEnvironment

__version__ = '1.0.0'
__all__ = [
    'TensorEvolution',
    'NeuralPopulation',
    'TensorLCIAgent',
    'VectorizedMarkovEnvironment',
]
