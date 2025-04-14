"""
Analysis Utilities

This module provides functions for analyzing LCI experiment results.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import json
import os
import logging
from scipy import stats
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


def calculate_summary_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a dataset
    
    Args:
        data: List of numeric values
        
    Returns:
        Dictionary of summary statistics
    """
    if not data:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None
        }
        
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75))
    }


def analyze_fitness_progression(fitness_history: List[float]) -> Dict[str, Any]:
    """
    Analyze fitness progression over generations
    
    Args:
        fitness_history: List of fitness values per generation
        
    Returns:
        Dictionary with analysis results
    """
    if not fitness_history or len(fitness_history) < 2:
        logger.warning("Insufficient data for fitness progression analysis")
        return {"error": "Insufficient data"}
    
    # Calculate overall statistics
    stats = calculate_summary_statistics(fitness_history)
    
    # Calculate improvement rate (linear regression slope)
    generations = np.arange(len(fitness_history))
    slope, intercept, r_value, p_value, std_err = stats.linregress(generations, fitness_history)
    
    # Calculate convergence (when fitness stabilizes)
    # Simple heuristic: when the improvement is less than 1% for 5 consecutive generations
    convergence_gen = None
    for i in range(5, len(fitness_history)):
        window = fitness_history[i-5:i]
        improvement = (fitness_history[i] - window[0]) / max(0.0001, window[0])
        if abs(improvement) < 0.01:
            convergence_gen = i - 5
            break
    
    # Calculate early vs late phase statistics
    early_phase = fitness_history[:len(fitness_history)//3]
    late_phase = fitness_history[-len(fitness_history)//3:]
    
    early_stats = calculate_summary_statistics(early_phase)
    late_stats = calculate_summary_statistics(late_phase)
    
    return {
        "overall_stats": stats,
        "early_phase_stats": early_stats,
        "late_phase_stats": late_stats,
        "improvement_rate": float(slope),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "convergence_generation": convergence_gen
    }


def analyze_lci_balance(agents_lci_balance: List[float]) -> Dict[str, Any]:
    """
    Analyze LCI balance distribution
    
    Args:
        agents_lci_balance: List of LCI balance values
        
    Returns:
        Dictionary with analysis results
    """
    if not agents_lci_balance:
        logger.warning("No data for LCI balance analysis")
        return {"error": "No data"}
    
    # Basic statistics
    stats = calculate_summary_statistics(agents_lci_balance)
    
    # Calculate skewness and kurtosis
    skewness = float(stats.skew(agents_lci_balance))
    kurtosis = float(stats.kurtosis(agents_lci_balance))
    
    # Test for normality
    shapiro_stat, shapiro_p = stats.shapiro(agents_lci_balance)
    
    # Calculate percentile ranks
    percentiles = {
        "10th": float(np.percentile(agents_lci_balance, 10)),
        "25th": float(np.percentile(agents_lci_balance, 25)),
        "50th": float(np.percentile(agents_lci_balance, 50)),
        "75th": float(np.percentile(agents_lci_balance, 75)),
        "90th": float(np.percentile(agents_lci_balance, 90))
    }
    
    return {
        "basic_stats": stats,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "normality_test": {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "is_normal": shapiro_p > 0.05
        },
        "percentiles": percentiles
    }


def analyze_lci_evolution(lci_history: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    """
    Analyze how LCI parameters evolve over generations
    
    Args:
        lci_history: List of (L, C, I) tuples per generation
        
    Returns:
        Dictionary with analysis results
    """
    if not lci_history:
        logger.warning("No data for LCI evolution analysis")
        return {"error": "No data"}
    
    # Convert to numpy array for easier manipulation
    lci_array = np.array(lci_history)
    
    # Extract separate L, C, I histories
    L_history = lci_array[:, 0]
    C_history = lci_array[:, 1]
    I_history = lci_array[:, 2]
    
    # Calculate statistics for each parameter
    L_stats = calculate_summary_statistics(L_history)
    C_stats = calculate_summary_statistics(C_history)
    I_stats = calculate_summary_statistics(I_history)
    
    # Calculate parameter trends
    generations = np.arange(len(lci_history))
    
    L_slope, _, L_r_value, L_p_value, _ = stats.linregress(generations, L_history)
    C_slope, _, C_r_value, C_p_value, _ = stats.linregress(generations, C_history)
    I_slope, _, I_r_value, I_p_value, _ = stats.linregress(generations, I_history)
    
    # Calculate balance convergence - how parameters converge to similar values
    # Lower values indicate more balanced L,C,I parameters
    balance_divergence = []
    for l, c, i in lci_history:
        # Calculate variance of normalized parameters
        params = np.array([l, c, i])
        if np.max(params) > 0:
            norm_params = params / np.max(params)
            divergence = np.var(norm_params)
            balance_divergence.append(divergence)
    
    balance_trend = None
    if balance_divergence:
        balance_slope, _, balance_r_value, balance_p_value, _ = stats.linregress(
            generations[-len(balance_divergence):], balance_divergence
        )
        balance_trend = {
            "slope": float(balance_slope),
            "r_squared": float(balance_r_value ** 2),
            "p_value": float(balance_p_value)
        }
    
    return {
        "L_stats": L_stats,
        "C_stats": C_stats,
        "I_stats": I_stats,
        "trends": {
            "L_trend": {
                "slope": float(L_slope),
                "r_squared": float(L_r_value ** 2),
                "p_value": float(L_p_value)
            },
            "C_trend": {
                "slope": float(C_slope),
                "r_squared": float(C_r_value ** 2),
                "p_value": float(C_p_value)
            },
            "I_trend": {
                "slope": float(I_slope),
                "r_squared": float(I_r_value ** 2),
                "p_value": float(I_p_value)
            }
        },
        "balance_trend": balance_trend
    }


def analyze_environmental_response(
    volatility_history: List[float],
    performance_history: List[float]
) -> Dict[str, Any]:
    """
    Analyze how agent performance responds to environmental volatility
    
    Args:
        volatility_history: List of volatility values over time
        performance_history: List of performance values over time
        
    Returns:
        Dictionary with analysis results
    """
    if not volatility_history or not performance_history:
        logger.warning("Missing data for environmental response analysis")
        return {"error": "Missing data"}
        
    if len(volatility_history) != len(performance_history):
        logger.warning("Volatility and performance histories must have the same length")
        return {"error": "Data length mismatch"}
    
    # Calculate correlation between volatility and performance
    correlation, p_value = stats.pearsonr(volatility_history, performance_history)
    
    # Analyze lag effects - how performance changes after volatility changes
    lag_correlations = []
    max_lag = min(10, len(volatility_history) // 5)  # Up to 10 lags or 1/5 of data length
    
    for lag in range(1, max_lag + 1):
        # Correlation between volatility and performance with a lag
        lagged_corr, lagged_p = stats.pearsonr(
            volatility_history[:-lag], 
            performance_history[lag:]
        )
        lag_correlations.append({
            "lag": lag,
            "correlation": float(lagged_corr),
            "p_value": float(lagged_p)
        })
    
    # Find optimal lag (strongest negative correlation)
    optimal_lag = min(lag_correlations, key=lambda x: x["correlation"])
    
    # Analyze stability during volatility changes
    stability_in_volatility = []
    
    for i in range(1, len(volatility_history)):
        volatility_change = abs(volatility_history[i] - volatility_history[i-1])
        if volatility_change > 0.1:  # Significant volatility change
            # Get performance before and after change
            before_idx = max(0, i-5)
            after_idx = min(len(performance_history)-1, i+5)
            
            before_perf = performance_history[before_idx:i]
            after_perf = performance_history[i:after_idx+1]
            
            if before_perf and after_perf:
                before_std = np.std(before_perf)
                after_std = np.std(after_perf)
                before_mean = np.mean(before_perf)
                after_mean = np.mean(after_perf)
                
                stability_in_volatility.append({
                    "time_step": i,
                    "volatility_change": float(volatility_change),
                    "before_std": float(before_std),
                    "after_std": float(after_std),
                    "performance_change": float(after_mean - before_mean)
                })
    
    return {
        "correlation": {
            "pearson_r": float(correlation),
            "p_value": float(p_value),
            "significant_correlation": p_value < 0.05
        },
        "lag_analysis": {
            "lag_correlations": lag_correlations,
            "optimal_lag": optimal_lag
        },
        "stability_analysis": {
            "volatility_change_points": len(stability_in_volatility),
            "stability_details": stability_in_volatility
        }
    }


def load_results(
    results_dir: str, 
    pattern: str = "stats_*.json"
) -> Dict[str, Any]:
    """
    Load experiment results from files
    
    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern for result files
        
    Returns:
        Dictionary with loaded results
    """
    results = {}
    
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return results
    
    # Find all matching files
    result_files = list(results_path.glob(pattern))
    if not result_files:
        logger.warning(f"No result files found matching pattern {pattern} in {results_dir}")
        return results
    
    # Load each file
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result_data = json.load(f)
                
            # Use the filename as the key (without extension)
            key = file_path.stem
            results[key] = result_data
            logger.debug(f"Loaded results from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load results from {file_path}: {e}")
    
    logger.info(f"Loaded {len(results)} result files from {results_dir}")
    return results


def export_analysis_to_csv(
    analysis_results: Dict[str, Any],
    output_file: str
):
    """
    Export analysis results to CSV format
    
    Args:
        analysis_results: Dictionary with analysis results
        output_file: Path to output CSV file
    """
    # Flatten nested dictionaries for CSV export
    flattened_data = {}
    
    def flatten_dict(d, prefix=""):
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flatten_dict(value, new_key)
            else:
                flattened_data[new_key] = value
    
    flatten_dict(analysis_results)
    
    # Convert to DataFrame and export
    df = pd.DataFrame([flattened_data])
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"Exported analysis results to {output_file}")
    except Exception as e:
        logger.error(f"Failed to export analysis results: {e}")


def perform_comprehensive_analysis(
    results_dir: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of experiment results
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save analysis outputs
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment results
    results = load_results(results_dir)
    if not results:
        logger.error("No results to analyze")
        return {"error": "No results to analyze"}
    
    # Use the most recent result for analysis
    latest_result_key = sorted(results.keys())[-1]
    result_data = results[latest_result_key]
    
    # Extract data for analysis
    fitness_history = result_data.get("fitness_history", [])
    lci_history = result_data.get("lci_history", [])
    best_agent_history = result_data.get("best_agent_history", [])
    
    # Perform analyses
    fitness_analysis = analyze_fitness_progression(fitness_history)
    lci_evolution_analysis = analyze_lci_evolution(lci_history)
    
    # Get LCI balance values from the best agents over time
    if best_agent_history:
        balance_values = [agent.get("lci_balance", 0) for agent in best_agent_history if "lci_balance" in agent]
        balance_analysis = analyze_lci_balance(balance_values)
    else:
        balance_analysis = {"error": "No balance data available"}
    
    # Comprehensive analysis results
    analysis_results = {
        "fitness_analysis": fitness_analysis,
        "lci_evolution_analysis": lci_evolution_analysis,
        "balance_analysis": balance_analysis,
        "metadata": {
            "source_result": latest_result_key,
            "analysis_date": pd.Timestamp.now().isoformat()
        }
    }
    
    # Export analysis to CSV
    export_analysis_to_csv(
        analysis_results,
        os.path.join(output_dir, "comprehensive_analysis.csv")
    )
    
    # Save detailed analysis results as JSON
    try:
        with open(os.path.join(output_dir, "comprehensive_analysis.json"), 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logger.info("Saved comprehensive analysis results to JSON")
    except Exception as e:
        logger.error(f"Failed to save analysis results to JSON: {e}")
    
    return analysis_results 