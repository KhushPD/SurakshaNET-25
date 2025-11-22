"""
Real-Time Visualization Service
================================
Generates visualizations from real-time metrics data.
Works with RealTimeMonitoringService to provide live dashboard updates.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from typing import Dict, List
import logging

from app.config import MULTICLASS_LABELS, BINARY_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeVisualizationService:
    """
    Real-time visualization service for dashboard plots.
    Generates plots from current metrics without requiring full prediction history.
    """
    
    def __init__(self):
        """Initialize with modern styling."""
        sns.set_style("darkgrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 11
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def plot_realtime_spider_chart(self, attack_type_counts: Dict[int, int]) -> str:
        """
        Real-time Spider/Radar Plot - Threat Vector Analysis.
        
        Args:
            attack_type_counts: Dictionary mapping attack type ID to count
            
        Returns:
            Base64 encoded PNG image
        """
        logger.debug("Generating real-time spider plot...")
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Extract counts for each category
        categories = [MULTICLASS_LABELS[i] for i in range(5)]
        counts = [attack_type_counts.get(i, 0) for i in range(5)]
        
        # Normalize to percentage
        total = sum(counts)
        values = [(c / total * 100) if total > 0 else 0 for c in counts]
        
        # Create angles
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = values + values[:1]
        angles = angles + angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color='#e74c3c', label='Threat Level')
        ax.fill(angles, values, alpha=0.25, color='#e74c3c')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11, weight='bold')
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)
        ax.set_title('Real-Time Threat Vector Analysis', 
                     size=16, weight='bold', pad=20, color='#2c3e50')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        return self._fig_to_base64(fig)
    
    def plot_realtime_pie_chart(self, attack_count: int, normal_count: int) -> str:
        """
        Real-time Pie Chart - Binary Classification Distribution.
        
        Args:
            attack_count: Number of attacks detected
            normal_count: Number of normal traffic
            
        Returns:
            Base64 encoded PNG image
        """
        logger.debug("Generating real-time pie chart...")
        
        fig, ax = plt.subplots(figsize=(9, 9))
        
        counts = [normal_count, attack_count]
        labels = ['Normal', 'Attack']
        colors = ['#27ae60', '#e74c3c']
        explode = (0.05, 0.1)
        
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_weight('bold')
        
        ax.set_title('Real-Time Binary Classification', 
                     fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
        
        return self._fig_to_base64(fig)
    
    def plot_realtime_timeline(self, timeline_data: Dict) -> str:
        """
        Real-time Timeline - Attack Rate Over Time.
        
        Args:
            timeline_data: Dictionary with timestamps and attack_rates
            
        Returns:
            Base64 encoded PNG image
        """
        logger.debug("Generating real-time timeline...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        timestamps = timeline_data.get("timestamps", [])
        attack_rates = timeline_data.get("attack_rates", [])
        
        if not timestamps or not attack_rates:
            # Empty plot with message
            ax.text(0.5, 0.5, 'No data available yet', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Real-Time Attack Detection Timeline', 
                        fontsize=16, weight='bold', pad=20, color='#2c3e50')
            return self._fig_to_base64(fig)
        
        # Use indices for x-axis (simpler than parsing timestamps)
        x = list(range(len(attack_rates)))
        
        ax.plot(x, attack_rates, linewidth=2.5, color='#e74c3c',
               marker='o', markersize=6, markerfacecolor='#c0392b',
               markeredgecolor='white', markeredgewidth=1.5, label='Attack Rate')
        
        ax.fill_between(x, attack_rates, alpha=0.3, color='#e74c3c')
        
        # Threshold line
        ax.axhline(y=50, color='#f39c12', linestyle='--', linewidth=2,
                  label='50% Threshold', alpha=0.7)
        
        ax.set_xlabel('Time Window', fontsize=12, weight='bold')
        ax.set_ylabel('Attack Rate (%)', fontsize=12, weight='bold')
        ax.set_title('Real-Time Attack Detection Timeline',
                    fontsize=16, weight='bold', pad=20, color='#2c3e50')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_ylim(-5, 105)
        
        return self._fig_to_base64(fig)
    
    def plot_realtime_attack_distribution(self, attack_type_counts: Dict[int, int]) -> str:
        """
        Real-time Attack Type Distribution - Bar Chart.
        
        Args:
            attack_type_counts: Dictionary mapping attack type ID to count
            
        Returns:
            Base64 encoded PNG image
        """
        logger.debug("Generating real-time attack distribution...")
        
        fig, ax = plt.subplots(figsize=(11, 7))
        
        categories = [MULTICLASS_LABELS[i] for i in range(5)]
        counts = [attack_type_counts.get(i, 0) for i in range(5)]
        colors = ['#27ae60', '#e67e22', '#3498db', '#9b59b6', '#e74c3c']
        
        bars = ax.barh(categories, counts, color=colors,
                      edgecolor='white', linewidth=2, height=0.7)
        
        # Add value labels
        max_count = max(counts) if counts else 1
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{int(count)}', ha='left', va='center', fontsize=11, weight='bold')
        
        ax.set_xlabel('Number of Instances', fontsize=12, weight='bold')
        ax.set_title('Real-Time Attack Type Distribution',
                    fontsize=16, weight='bold', pad=20, color='#2c3e50')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        return self._fig_to_base64(fig)
    
    def plot_realtime_confidence(self, confidence_data: Dict) -> str:
        """
        Real-time Prediction Confidence - Histogram.
        
        Args:
            confidence_data: Dictionary with bin_edges, counts, and mean
            
        Returns:
            Base64 encoded PNG image
        """
        logger.debug("Generating real-time confidence plot...")
        
        fig, ax = plt.subplots(figsize=(11, 6))
        
        bin_edges = confidence_data.get("bin_edges", [])
        counts = confidence_data.get("counts", [])
        mean_conf = confidence_data.get("mean", 0.0)
        
        if not bin_edges or not counts:
            ax.text(0.5, 0.5, 'No data available yet',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Real-Time Prediction Confidence',
                        fontsize=16, weight='bold', pad=20, color='#2c3e50')
            return self._fig_to_base64(fig)
        
        # Plot histogram bars
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 0.05
        
        bars = ax.bar(bin_centers, counts, width=bin_width * 0.9, 
                     edgecolor='white', linewidth=1.5, alpha=0.7)
        
        # Color code by confidence level
        for bar, center in zip(bars, bin_centers):
            if center < 0.5:
                bar.set_facecolor('#e74c3c')  # Low - red
            elif center < 0.75:
                bar.set_facecolor('#f39c12')  # Medium - orange
            else:
                bar.set_facecolor('#27ae60')  # High - green
        
        # Mean line
        if mean_conf > 0:
            ax.axvline(mean_conf, color='#2c3e50', linestyle='--', linewidth=2.5,
                      label=f'Mean: {mean_conf:.3f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')
        ax.set_title('Real-Time Prediction Confidence',
                    fontsize=16, weight='bold', pad=20, color='#2c3e50')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        if mean_conf > 0:
            ax.legend(fontsize=11)
        ax.set_xlim(0, 1)
        
        return self._fig_to_base64(fig)
    
    def plot_realtime_binary_metrics(self, attack_count: int, normal_count: int) -> str:
        """
        Real-time Binary Classification Metrics - Enhanced View.
        
        Args:
            attack_count: Number of attacks
            normal_count: Number of normal traffic
            
        Returns:
            Base64 encoded PNG image
        """
        logger.debug("Generating real-time binary metrics...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        labels = ['Normal', 'Attack']
        counts = [normal_count, attack_count]
        colors = ['#27ae60', '#e74c3c']
        
        total = sum(counts)
        
        # Left: Bar chart
        bars = ax1.bar(labels, counts, color=colors, edgecolor='white',
                      linewidth=3, width=0.6)
        
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total * 100) if total > 0 else 0
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=12, weight='bold')
        
        ax1.set_ylabel('Count', fontsize=12, weight='bold')
        ax1.set_title('Classification Counts', fontsize=14, weight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Right: Donut chart
        if total > 0:
            wedges, texts, autotexts = ax2.pie(
                counts,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3),
                textprops={'fontsize': 11, 'weight': 'bold'}
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(13)
        else:
            ax2.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
        
        ax2.set_title('Percentage Distribution', fontsize=14, weight='bold')
        
        fig.suptitle('Real-Time Binary Classification Metrics',
                    fontsize=16, weight='bold', y=1.02, color='#2c3e50')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_realtime_plots(self, metrics: Dict) -> Dict[str, str]:
        """
        Generate all real-time plots from current metrics.
        
        Args:
            metrics: Current metrics dictionary from RealTimeMetrics
            
        Returns:
            Dictionary mapping plot names to base64 images
        """
        logger.info("Generating real-time visualization plots...")
        
        plots = {}
        
        try:
            # Extract metrics
            attack_type_counts = metrics.get("attack_type_counts_recent", {})
            recent_attack = metrics.get("recent_attack_count", 0)
            recent_normal = metrics.get("recent_normal_count", 0)
            
            # Generate plots
            plots["spider_plot"] = self.plot_realtime_spider_chart(attack_type_counts)
            plots["pie_chart"] = self.plot_realtime_pie_chart(recent_attack, recent_normal)
            plots["attack_type_distribution"] = self.plot_realtime_attack_distribution(attack_type_counts)
            plots["binary_classification"] = self.plot_realtime_binary_metrics(recent_attack, recent_normal)
            
            logger.info(f"✅ Generated {len(plots)} real-time plots successfully")
            
        except Exception as e:
            logger.error(f"❌ Error generating real-time plots: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return plots
    
    def generate_realtime_plots_with_timeline(self, metrics: Dict, 
                                             timeline_data: Dict,
                                             confidence_data: Dict) -> Dict[str, str]:
        """
        Generate all real-time plots including timeline and confidence.
        
        Args:
            metrics: Current metrics dictionary
            timeline_data: Timeline data from get_timeline()
            confidence_data: Confidence distribution from get_confidence_distribution()
            
        Returns:
            Dictionary mapping plot names to base64 images
        """
        logger.info("Generating complete real-time visualization suite...")
        
        plots = self.generate_realtime_plots(metrics)
        
        try:
            # Add timeline and confidence plots
            plots["timeline"] = self.plot_realtime_timeline(timeline_data)
            plots["prediction_confidence"] = self.plot_realtime_confidence(confidence_data)
            
            logger.info(f"✅ Generated {len(plots)} plots with timeline and confidence")
            
        except Exception as e:
            logger.error(f"❌ Error generating timeline/confidence plots: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return plots


# Global real-time visualization service
realtime_viz_service = RealTimeVisualizationService()
