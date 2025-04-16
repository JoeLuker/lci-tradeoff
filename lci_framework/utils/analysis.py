"""
Analysis Utilities

This module provides functions for analyzing LCI experiment results.
"""

import mlx.core as mx
from typing import List, Dict, Tuple, Any, Optional
import json
import os
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


def calculate_summary_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a dataset using MLX
    
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
    
    # Convert to MLX array
    data_mx = mx.array(data)
    
    # Calculate basic statistics
    mean = mx.mean(data_mx).item()
    std = mx.std(data_mx).item()
    min_val = mx.min(data_mx).item()
    max_val = mx.max(data_mx).item()
    
    # Sort data for percentiles
    sorted_data = mx.sort(data_mx)
    n = len(data)
    
    # Calculate median (50th percentile)
    if n % 2 == 0:
        # Even number of elements
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]).item() / 2
    else:
        # Odd number of elements
        median = sorted_data[n//2].item()
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    q1 = sorted_data[q1_idx].item()
    q3 = sorted_data[q3_idx].item()
    
    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "min": float(min_val),
        "max": float(max_val),
        "q1": float(q1),
        "q3": float(q3)
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
    # y = mx + b
    # Use MLX for linear regression calculations
    generations = mx.array(range(len(fitness_history)))
    fitness_mx = mx.array(fitness_history)
    
    n = len(fitness_history)
    mean_x = mx.mean(generations).item()
    mean_y = mx.mean(fitness_mx).item()
    
    # Calculate slope: m = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
    numerator = mx.sum((generations - mean_x) * (fitness_mx - mean_y)).item()
    denominator = mx.sum((generations - mean_x) ** 2).item()
    
    if denominator != 0:
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        
        # Calculate r^2
        y_pred = slope * generations + intercept
        ss_total = mx.sum((fitness_mx - mean_y) ** 2).item()
        ss_residual = mx.sum((fitness_mx - y_pred) ** 2).item()
        
        if ss_total != 0:
            r_squared = 1 - (ss_residual / ss_total)
        else:
            r_squared = 0
    else:
        slope = 0
        intercept = mean_y
        r_squared = 0
    
    # Calculate convergence (when fitness stabilizes)
    # Simple heuristic: when the improvement is less than 1% for 5 consecutive generations
    convergence_gen = None
    for i in range(5, len(fitness_history)):
        window = mx.array(fitness_history[i-5:i])
        window_start = window[0].item()
        current = fitness_history[i]
        
        if window_start != 0:
            improvement = (current - window_start) / abs(window_start)
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
        "r_squared": float(r_squared),
        "convergence_generation": convergence_gen
    }


def analyze_lci_balance(agents_lci_balance: List[float]) -> Dict[str, Any]:
    """
    Analyze LCI balance distribution using MLX
    
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
    
    # Convert to MLX array
    balance_mx = mx.array(agents_lci_balance)
    
    # Calculate mean, variance for further calculations
    mean = mx.mean(balance_mx).item()
    var = mx.var(balance_mx).item()
    n = len(agents_lci_balance)
    
    # Calculate skewness (3rd moment)
    # skewness = E[(X - μ)³] / σ³
    if var > 0:
        m3 = mx.mean((balance_mx - mean) ** 3).item()
        skewness = m3 / (var ** 1.5)
    else:
        skewness = 0.0
    
    # Calculate kurtosis (4th moment)
    # kurtosis = E[(X - μ)⁴] / σ⁴ - 3
    if var > 0:
        m4 = mx.mean((balance_mx - mean) ** 4).item()
        kurtosis = m4 / (var ** 2) - 3
    else:
        kurtosis = 0.0
    
    # We can't do a Shapiro-Wilk test with MLX directly
    # Instead, we'll use a simpler normality check based on skewness and kurtosis
    is_normal = abs(skewness) < 1.0 and abs(kurtosis) < 1.0
    
    # Calculate percentile ranks using sorted array
    sorted_balance = mx.sort(balance_mx)
    
    percentiles = {
        "10th": float(sorted_balance[int(n * 0.1)].item()),
        "25th": float(sorted_balance[int(n * 0.25)].item()),
        "50th": float(sorted_balance[int(n * 0.5)].item()),
        "75th": float(sorted_balance[int(n * 0.75)].item()),
        "90th": float(sorted_balance[int(n * 0.9)].item())
    }
    
    return {
        "basic_stats": stats,
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "normality_check": {
            "is_normal": is_normal,
            "skewness": float(skewness),
            "kurtosis": float(kurtosis)
        },
        "percentiles": percentiles
    }


def analyze_lci_evolution(lci_history: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    """
    Analyze how LCI parameters evolve over generations using MLX
    
    Args:
        lci_history: List of (L, C, I) tuples per generation
        
    Returns:
        Dictionary with analysis results
    """
    if not lci_history:
        logger.warning("No data for LCI evolution analysis")
        return {"error": "No data"}
    
    # Convert to MLX array for easier manipulation
    lci_array = mx.array(lci_history)
    
    # Extract separate L, C, I histories
    L_history = lci_array[:, 0]
    C_history = lci_array[:, 1]
    I_history = lci_array[:, 2]
    
    # Calculate statistics for each parameter
    L_stats = calculate_summary_statistics([l.item() for l in L_history])
    C_stats = calculate_summary_statistics([c.item() for c in C_history])
    I_stats = calculate_summary_statistics([i.item() for i in I_history])
    
    # Calculate parameter trends using linear regression
    generations = mx.array(range(len(lci_history)))
    
    # Calculate trends for L
    l_slope, l_r_squared = _calculate_linear_trend(generations, L_history)
    
    # Calculate trends for C
    c_slope, c_r_squared = _calculate_linear_trend(generations, C_history)
    
    # Calculate trends for I
    i_slope, i_r_squared = _calculate_linear_trend(generations, I_history)
    
    # Calculate balance convergence - how parameters converge to similar values
    # Lower values indicate more balanced L,C,I parameters
    balance_divergence = []
    for gen_idx in range(len(lci_history)):
        # Get L, C, I for this generation
        l, c, i = lci_array[gen_idx].tolist()
        
        # Calculate variance of normalized parameters
        params = mx.array([l, c, i])
        if mx.max(params) > 0:
            norm_params = params / mx.max(params)
            divergence = mx.var(norm_params).item()
            balance_divergence.append(divergence)
    
    balance_trend = None
    if balance_divergence:
        balance_gen = mx.array(range(len(balance_divergence)))
        balance_div = mx.array(balance_divergence)
        balance_slope, balance_r_squared = _calculate_linear_trend(balance_gen, balance_div)
        
        balance_trend = {
            "slope": float(balance_slope),
            "r_squared": float(balance_r_squared)
        }
    
    return {
        "L_stats": L_stats,
        "C_stats": C_stats,
        "I_stats": I_stats,
        "trends": {
            "L_trend": {
                "slope": float(l_slope),
                "r_squared": float(l_r_squared)
            },
            "C_trend": {
                "slope": float(c_slope),
                "r_squared": float(c_r_squared)
            },
            "I_trend": {
                "slope": float(i_slope),
                "r_squared": float(i_r_squared)
            }
        },
        "balance_trend": balance_trend
    }


def _calculate_linear_trend(x: mx.array, y: mx.array) -> Tuple[float, float]:
    """
    Calculate linear regression using MLX
    
    Args:
        x: Independent variable
        y: Dependent variable
        
    Returns:
        Tuple of (slope, r_squared)
    """
    mean_x = mx.mean(x).item()
    mean_y = mx.mean(y).item()
    
    # Calculate slope: m = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
    numerator = mx.sum((x - mean_x) * (y - mean_y)).item()
    denominator = mx.sum((x - mean_x) ** 2).item()
    
    if denominator != 0:
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        
        # Calculate r^2
        y_pred = slope * x + intercept
        ss_total = mx.sum((y - mean_y) ** 2).item()
        ss_residual = mx.sum((y - y_pred) ** 2).item()
        
        if ss_total != 0:
            r_squared = 1 - (ss_residual / ss_total)
        else:
            r_squared = 0
    else:
        slope = 0
        r_squared = 0
    
    return slope, r_squared


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
    
    # Convert to MLX arrays
    vol_mx = mx.array(volatility_history)
    perf_mx = mx.array(performance_history)
    
    # Calculate correlation
    # r = cov(X,Y) / (std(X) * std(Y))
    mean_vol = mx.mean(vol_mx).item()
    mean_perf = mx.mean(perf_mx).item()
    
    # Calculate covariance
    cov = mx.mean((vol_mx - mean_vol) * (perf_mx - mean_perf)).item()
    
    # Calculate standard deviations
    std_vol = mx.std(vol_mx).item()
    std_perf = mx.std(perf_mx).item()
    
    # Calculate correlation
    if std_vol > 0 and std_perf > 0:
        correlation = cov / (std_vol * std_perf)
    else:
        correlation = 0
    
    # Calculate lag correlation (performance vs volatility with time lag)
    # For simplicity, we'll check lags of 1-5 time steps
    lag_correlations = []
    
    for lag in range(1, 6):
        if lag < len(volatility_history):
            # Align volatility with lagged performance
            lagged_vol = vol_mx[:-lag]
            lagged_perf = perf_mx[lag:]
            
            if len(lagged_vol) > 1:
                # Calculate correlation with lag
                mean_lagged_vol = mx.mean(lagged_vol).item()
                mean_lagged_perf = mx.mean(lagged_perf).item()
                
                cov_lag = mx.mean((lagged_vol - mean_lagged_vol) * 
                                  (lagged_perf - mean_lagged_perf)).item()
                
                std_lagged_vol = mx.std(lagged_vol).item()
                std_lagged_perf = mx.std(lagged_perf).item()
                
                if std_lagged_vol > 0 and std_lagged_perf > 0:
                    lag_corr = cov_lag / (std_lagged_vol * std_lagged_perf)
                else:
                    lag_corr = 0
                
                lag_correlations.append({"lag": lag, "correlation": float(lag_corr)})
    
    # Analyze effect of volatility changes
    # Find periods of significant volatility increase/decrease
    volatility_changes = []
    for i in range(1, len(volatility_history)):
        change = volatility_history[i] - volatility_history[i-1]
        if abs(change) > 0.1:  # Threshold for significant change
            volatility_changes.append({"index": i, "change": change})
    
    # Analyze performance before and after volatility changes
    change_analyses = []
    window_size = 3  # Look at performance 3 steps before and after
    
    for change in volatility_changes:
        idx = change["index"]
        
        # Get performance before and after
        before_start = max(0, idx - window_size)
        after_end = min(len(performance_history), idx + window_size + 1)
        
        before_perf = mx.array(performance_history[before_start:idx])
        after_perf = mx.array(performance_history[idx:after_end])
        
        if len(before_perf) > 0 and len(after_perf) > 0:
            # Calculate statistics
            before_std = mx.std(before_perf).item()
            after_std = mx.std(after_perf).item()
            before_mean = mx.mean(before_perf).item()
            after_mean = mx.mean(after_perf).item()
            
            # Calculate performance change
            perf_change = after_mean - before_mean
            variance_change = after_std - before_std
            
            change_analyses.append({
                "index": idx,
                "volatility_change": change["change"],
                "performance_before": float(before_mean),
                "performance_after": float(after_mean),
                "performance_change": float(perf_change),
                "variance_before": float(before_std),
                "variance_after": float(after_std),
                "variance_change": float(variance_change)
            })
    
    return {
        "volatility_performance_correlation": float(correlation),
        "lag_correlations": lag_correlations,
        "volatility_change_analyses": change_analyses
    }


def load_results(
    results_dir: str, 
    pattern: str = "stats_*.json"
) -> Dict[str, Any]:
    """
    Load experimental results from JSON files
    
    Args:
        results_dir: Directory containing results
        pattern: Glob pattern for result files
        
    Returns:
        Dictionary of results
    """
    results = {}
    
    try:
        # Find all matching files
        result_files = list(Path(results_dir).glob(pattern))
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Use the filename without extension as the key
                key = file_path.stem
                results[key] = data
                
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                
    except Exception as e:
        logger.error(f"Error reading results directory: {e}")
    
    return results


def export_analysis_to_csv(
    analysis_results: Dict[str, Any],
    output_file: str
):
    """
    Export analysis results to CSV format
    
    Args:
        analysis_results: Dictionary of analysis results
        output_file: Output CSV file path
    """
    try:
        # Flatten nested dictionary
        flat_data = flatten_dict(analysis_results)
        
        # Write to CSV file
        with open(output_file, 'w') as f:
            # Write header
            f.write(','.join(flat_data.keys()) + '\n')
            
            # Write values
            f.write(','.join(str(v) for v in flat_data.values()) + '\n')
            
        logger.info(f"Exported analysis results to {output_file}")
        
    except Exception as e:
        logger.error(f"Error exporting analysis results: {e}")
        
    def flatten_dict(d, prefix=""):
        """Flatten nested dictionary structure"""
        result = {}
        
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(flatten_dict(value, new_key))
            else:
                result[new_key] = value
                
        return result


def perform_comprehensive_analysis(
    results_dir: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on experiment results
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save analysis outputs
        
    Returns:
        Dictionary of analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(results_dir)
    
    analysis_results = {}
    
    for exp_name, exp_data in results.items():
        # Extract data for analysis
        exp_analysis = {}
        
        # Analyze fitness progression if available
        if "fitness_history" in exp_data:
            exp_analysis["fitness_progression"] = analyze_fitness_progression(
                exp_data["fitness_history"]
            )
        
        # Analyze LCI evolution if available
        if "lci_history" in exp_data:
            exp_analysis["lci_evolution"] = analyze_lci_evolution(
                exp_data["lci_history"]
            )
        
        # Analyze environmental response if available
        if "volatility_history" in exp_data and "performance_history" in exp_data:
            exp_analysis["environmental_response"] = analyze_environmental_response(
                exp_data["volatility_history"],
                exp_data["performance_history"]
            )
        
        # Store analysis results
        analysis_results[exp_name] = exp_analysis
        
        # Export to JSON
        output_file = os.path.join(output_dir, f"{exp_name}_analysis.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(exp_analysis, f, indent=2)
            logger.info(f"Saved analysis for {exp_name} to {output_file}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    # Export summary to CSV
    export_analysis_to_csv(
        analysis_results,
        os.path.join(output_dir, "analysis_summary.csv")
    )
    
    return analysis_results 