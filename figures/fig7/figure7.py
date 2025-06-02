import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
from matplotlib.axes import Axes
from PIL import Image
from typing import List, Dict, Any

font_dict = {'size': 12}
plot_style = {'color': '#3A3A3A', 'alpha':0.8, 's':30}

def load_data(data_path:str)->List[Any]:
    return list(map(lambda x:json.loads(x),open(data_path, 'r').read().splitlines()))


# a)
def plot_log_log_entities(records:List[Any], ax:Axes):
    """
    Creates a log-log plot of entity counts.
    """
    # number of unique mentions to the same entity in the input file
    entity_counts = [len(record['entity_references']) for record in records]

    # Count frequency of each count
    freq_counter = Counter(entity_counts)
    
    # Prepare data
    x = np.array(sorted(freq_counter.keys()))
    y = np.array([freq_counter[k] for k in x])
    
    # Scatter plot with log scales
    ax.scatter(x, y, **plot_style)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel("Number of Queries per Wikipedia URI", **font_dict)
    ax.set_ylabel("Number of Wikipedia URIs", **font_dict)
    
    plt.setp(ax.get_xticklabels(), **font_dict)
    plt.setp(ax.get_yticklabels(), **font_dict)
    
    ax.grid(True, which="major", linestyle="dashed", linewidth=0.3, alpha=0.7)  
    ax.grid(True, which="minor", linestyle="dotted", linewidth=0.3, alpha=0.5)  

# b)
def plot_leader_counts_by_language(records:List[Any], ax:Axes):
    """
    Creates a log-log plot of leader counts by language count instead of bar chart.
    """
    # Extract language counts
    lang_counts = [record.get('language_count', 0) for record in records]
    
    # Count frequency of each language count
    lang_counter = Counter(lang_counts)
    
    # Prepare data
    x = np.array(sorted(lang_counter.keys()))
    y = np.array([lang_counter[k] for k in x])
    
    # Scatter plot with log scales - using koyu gri renk
    ax.scatter(x, y, **plot_style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel("Number of WikiProjects per Wikipedia URI", **font_dict)
    ax.set_ylabel("Number of Wikipedia URIs", **font_dict)
    
    plt.setp(ax.get_xticklabels(), **font_dict)
    plt.setp(ax.get_yticklabels(), **font_dict)
    
    ax.grid(True, which="major", linestyle="dashed", linewidth=0.3, alpha=0.7)  
    ax.grid(True, which="minor", linestyle="dotted", linewidth=0.3, alpha=0.5)  

# c)
def plot_word_count_distribution(records:List[Any], ax:Axes,num:int=30):
    """
    Creates a log-log plot of article counts by word count.
    """
    # Extract word counts
    word_counts = [record.get('word_count', 0) for record in records]
    
    # Filter out invalid word counts (negative values or errors)
    word_counts = [count for count in word_counts if count > 0]
    
    if not word_counts:
        print("Warning: No valid word counts found")
        return
    
    # Group word counts into bins (for better visualization)
    # Using logarithmic bins for log-log plot
    min_count = min(word_counts)
    max_count = max(word_counts)
    
    # Create bins in log space
    bins = np.logspace(np.log10(max(1, min_count)), np.log10(max_count + 1), num)
    
    # Count frequency in each bin
    hist, bin_edges = np.histogram(word_counts, bins=bins)
    
    # Use bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter out empty bins
    mask = hist > 0
    x = bin_centers[mask]
    y = hist[mask]
 
    ax.scatter(x, y, **plot_style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel("Number of WikiProjects per Wikipedia URI", **font_dict)
    ax.set_ylabel("Number of Wikipedia URIs", **font_dict)
    
    plt.setp(ax.get_xticklabels(), **font_dict)
    plt.setp(ax.get_yticklabels(), **font_dict)
    
    ax.grid(True, which="major", linestyle="dashed", linewidth=0.3, alpha=0.7)  
    ax.grid(True, which="minor", linestyle="dotted", linewidth=0.3, alpha=0.5)  
