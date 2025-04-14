# LCI Framework

A framework for evolutionary experiments on the trade-offs between Losslessness, Compression, and Invariance (LCI) in neural agents.

## Overview

The LCI Framework is a research tool for exploring how neural agents balance three fundamental properties:

- **L (Losslessness)**: The ability to retain and use information without loss.
- **C (Compression)**: The ability to efficiently encode information in a compact representation.
- **I (Invariance)**: The ability to maintain stable representations despite environmental changes.

These properties form a trade-off triangle similar to other trilemmas in mathematics, computer science, and economics.

This framework allows researchers to:
1. Create neural agents with different LCI parameter settings
2. Test these agents in environments with controllable volatility
3. Use evolutionary optimization to discover optimal LCI balance for various conditions
4. Analyze the relationship between environmental volatility and optimal LCI parameters

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- NumPy, Matplotlib, and other scientific libraries

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lci-framework.git
cd lci-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests to verify installation:
```bash
python -m unittest discover
```

## Usage

### Quick Start

To run a simple LCI experiment:

```bash
python -m lci_framework.main run
```

This will run an evolutionary simulation with default parameters and save the results to the `results` directory.

### Command Line Interface

The framework provides a command-line interface with the following commands:

1. **Initialize a configuration file**:
```bash
python -m lci_framework.main init --output config.yaml
```

2. **Run an experiment with a custom configuration**:
```bash
python -m lci_framework.main run --config config.yaml --output-dir results --name my_experiment
```

3. **Analyze experiment results**:
```bash
python -m lci_framework.main analyze --results-dir results/my_experiment --output-dir analysis
```

4. **Visualize experiment results**:
```bash
python -m lci_framework.main visualize --results-dir results/my_experiment --output-dir visualizations
```

### Configuration

The configuration file uses YAML or JSON format and includes settings for:

- Environment parameters (number of states, volatility schedule)
- Evolution parameters (population size, mutation rate, etc.)
- Agent parameters (neural network architecture, energy budget)

Example configuration:

```yaml
environment:
  n_states: 20
  sparse_transitions: true
  seed: 42

evolution:
  pop_size: 50
  mutation_rate: 0.1
  tournament_size: 5
  elitism: 2
  n_generations: 50
  steps_per_eval: 100
  seed: 42

agent:
  learning_rate_range: [0.001, 0.1]
  hidden_size_range: [8, 128]
  n_layers_range: [1, 3]
  l1_reg_range: [0.0, 0.01]
  l2_reg_range: [0.0, 0.01]
  dropout_rate_range: [0.0, 0.5]
  initial_energy: 1000.0

output:
  output_dir: results
  save_interval: 10
```

## Core Components

### Markov Environment

The environment simulates a Markov process with configurable volatility. Volatility represents how frequently the transition probabilities between states change. This allows testing how agents adapt to stable vs. changing environments.

```python
from lci_framework.environments.markov_environment import MarkovEnvironment

# Create environment with 20 states
env = MarkovEnvironment(n_states=20, sparse_transitions=True)

# Set a custom volatility schedule
volatility_schedule = [
    (1000, 0.0),  # 1000 steps of stability
    (1000, 0.5),  # 1000 steps of medium volatility
    (1000, 0.0),  # 1000 steps of stability
]
env.set_volatility_schedule(volatility_schedule)

# Reset environment to initial state
state = env.reset()
observation = env.get_observation(state)
```

### LCI Agent

Agents use neural networks to predict the next state based on the current observation. The LCI balance is controlled through model parameters:

```python
from lci_framework.agents.lci_agent import LCIAgent

# Create agent with specific LCI parameters
agent = LCIAgent(
    input_size=20,           
    learning_rate=0.01,      # L parameter
    hidden_size=64,          # C parameter (inverse)
    n_layers=2,              # C parameter (inverse)
    l1_reg=0.001,            # I parameter
    l2_reg=0.002,            # I parameter
    dropout_rate=0.1,        # I parameter
)

# Get LCI balance score (0 to 1, higher = more balanced)
balance = agent.get_lci_balance()
print(f"LCI Balance: {balance:.3f}")
```

### Evolution Framework

The framework uses evolutionary optimization to find optimal LCI parameters:

```python
from lci_framework.core.evolution import LCIEvolution

# Environment configuration
env_config = {"n_states": 20, "sparse_transitions": True}

# Create evolution framework
evolution = LCIEvolution(
    env_config=env_config,
    pop_size=50,
    mutation_rate=0.1,
    tournament_size=5,
    elitism=2,
    output_dir="results"
)

# Run simulation
evolution.run_simulation(n_generations=50, steps_per_eval=100)

# Plot results
evolution.plot_results()
```

## Advanced Usage

### Custom Volatility Patterns

You can create custom volatility patterns to test specific hypotheses:

```python
# Oscillating volatility
steps_per_cycle = 500
n_cycles = 10
volatility_schedule = []

for i in range(n_cycles):
    volatility_schedule.append((steps_per_cycle // 2, 0.0))  # Stability
    volatility_schedule.append((steps_per_cycle // 2, 0.8))  # High volatility

env.set_volatility_schedule(volatility_schedule)
```

### Analyzing LCI Balance

The framework provides tools to analyze the relationship between LCI balance and agent performance:

```python
from lci_framework.utils.analysis import analyze_lci_balance, analyze_lci_evolution

# Get LCI balance values from population
balance_values = [agent.get_lci_balance() for agent in evolution.population]

# Analyze balance distribution
balance_analysis = analyze_lci_balance(balance_values)
print(f"Mean balance: {balance_analysis['basic_stats']['mean']:.3f}")

# Analyze how LCI parameters evolved over generations
lci_evolution_analysis = analyze_lci_evolution(evolution.lci_history)
```

## Project Structure

```
lci_framework/
├── agents/                  # Agent implementations
│   └── lci_agent.py         # LCI agent class
├── core/                    # Core framework modules
│   └── evolution.py         # Evolutionary optimization
├── environments/            # Environment implementations
│   └── markov_environment.py # Markov process environment
├── utils/                   # Utility modules
│   ├── analysis.py          # Analysis utilities
│   ├── config.py            # Configuration utilities
│   └── visualization.py     # Visualization utilities
├── tests/                   # Test modules
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
└── main.py                  # Main entry point
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```
@misc{lci-framework,
  author = {Your Name},
  title = {LCI Framework: A Testbed for Exploring Losslessness-Compression-Invariance Trade-offs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/lci-framework}}
}
```

## Acknowledgments

This framework was inspired by the work on information bottleneck theory, predictive coding, and meta-learning for non-stationary environments. 