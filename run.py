#!/usr/bin/env python3
"""
Runner script for LCI framework with tensor-based implementation.
"""

import os
import sys
import argparse
from lci_framework.main import main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LCI simulation")
    parser.add_argument("--config", type=str, default="config/lci_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, 
                        help="Output directory (overrides config setting if provided)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Run the simulation
    main(args) 