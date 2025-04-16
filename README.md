# GPU-Accelerated LCI Framework

This is a tensor-based implementation of the LCI (Learning, Cognition, Intelligence) framework that leverages GPU acceleration through PyTorch for vastly improved performance.

## Key Features

- **GPU Acceleration**: Utilizes MPS (Metal Performance Shaders) on Apple Silicon, CUDA on NVIDIA GPUs, and falls back to CPU when needed
- **Vectorized Operations**: Uses tensor operations for batch processing of environment states and agent actions
- **Population-based Neural Networks**: Efficiently processes entire populations of neural networks simultaneously
- **High Performance**: Achieves orders of magnitude speedup compared to traditional implementations

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (for NVIDIA acceleration) or Apple Silicon Mac (for MPS acceleration)
- Other dependencies listed in `requirements.txt`

## Usage

### Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run a simulation:

```bash
python run.py
```

### Configuration

The implementation uses YAML configuration files. A sample is provided in `config/lci_config.yaml`.

Key parameters:

- `pop_size`: Population size for evolutionary algorithm
- `n_states`: Number of states in the environment
- `n_actions`: Number of possible actions
- `hidden_size`: Size of hidden layers in neural networks
- `n_layers`: Number of hidden layers
- `mutation_rate`: Rate of mutation for genetic algorithm
- `n_generations`: Number of evolutionary generations to run
- `steps_per_generation`: Number of environmental steps per generation

### Command Line Options

```bash
python run.py --config config/lci_config.yaml --output results/my_run --log-level INFO
```

Options:

- `--config`: Path to configuration file
- `--output`: Output directory (overrides config setting)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Architecture Overview

The implementation consists of these key components:

1. **Vectorized Environment** (`VectorizedMarkovEnvironment`): Processes environment transitions for all agents simultaneously using tensor operations

2. **Neural Population** (`NeuralPopulation`): Manages a population of neural networks that share architecture but have individual weights, enabling parallel forward and backward passes

3. **Tensor Evolution** (`TensorEvolution`): Implements evolutionary algorithms using tensor operations for selection, crossover, and fitness evaluation
