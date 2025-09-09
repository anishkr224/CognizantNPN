# Evaluation Metrics for AI Revenue Leakage Detection System

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationMetrics:
    """
    Class for evaluating the performance of the AI Revenue Leakage Detection System.
    Provides metrics and visualizations to assess the accuracy and effectiveness
    of the revenue leakage detection.
    """
    
    def __init__(self):
        """
        Initialize the EvaluationMetrics class.
        """
        self.results = None
        self.ground_truth = None
        
    def load_data(self, results_df, ground_truth_df):
        """
        Load the results from the AI system and the ground truth data.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing the detection results
            ground_truth_df (pd.DataFrame): DataFrame containing the ground truth
        """
        self.results = results_df
        self.ground_truth = ground_truth_df
        
    def calculate_metrics(self):
        """
        Calculate precision, recall, F1-score, and other relevant metrics.
        
        Returns:
            dict: Dictionary containing the calculated metrics
        """
        if self.results is None or self.ground_truth is None:
            raise ValueError("Results and ground truth data must be loaded first")
        
        # Extract true and predicted labels
        y_true = self.ground_truth['is_leakage'].values
        y_pred = self.results['is_detected'].values
        
        # Calculate precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate financial impact metrics
        if 'amount' in self.ground_truth.columns and 'detected_amount' in self.results.columns:
            true_leakage_amount = self.ground_truth[self.ground_truth['is_leakage']]['amount'].sum()
            detected_leakage_amount = self.results[self.results['is_detected']]['detected_amount'].sum()
            recovery_rate = detected_leakage_amount / true_leakage_amount if true_leakage_amount > 0 else 0
        else:
            true_leakage_amount = 0
            detected_leakage_amount = 0
            recovery_rate = 0
        
        # Compile metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'true_leakage_amount': true_leakage_amount,
            'detected_leakage_amount': detected_leakage_amount,
            'recovery_rate': recovery_rate
        }
        
        return metrics
    
    def visualize_confusion_matrix(self):
        """
        Generate a visualization of the confusion matrix.
        
        Returns:
            matplotlib.figure.Figure: The confusion matrix visualization
        """
        if self.results is None or self.ground_truth is None:
            raise ValueError("Results and ground truth data must be loaded first")
        
        # Extract true and predicted labels
        y_true = self.ground_truth['is_leakage'].values
        y_pred = self.results['is_detected'].values
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure and plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Leakage', 'Leakage'],
                   yticklabels=['No Leakage', 'Leakage'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        return plt.gcf()
    
    def visualize_financial_impact(self):
        """
        Generate a visualization of the financial impact of detected vs. actual leakage.
        
        Returns:
            matplotlib.figure.Figure: The financial impact visualization
        """
        if self.results is None or self.ground_truth is None:
            raise ValueError("Results and ground truth data must be loaded first")
        
        if 'amount' not in self.ground_truth.columns or 'detected_amount' not in self.results.columns:
            raise ValueError("Ground truth and results must contain amount columns")
        
        # Calculate financial metrics
        true_leakage_amount = self.ground_truth[self.ground_truth['is_leakage']]['amount'].sum()
        detected_leakage_amount = self.results[self.results['is_detected']]['detected_amount'].sum()
        missed_leakage_amount = true_leakage_amount - detected_leakage_amount
        
        # Create figure and plot financial impact
        plt.figure(figsize=(10, 6))
        
        # Bar chart
        categories = ['True Leakage', 'Detected Leakage', 'Missed Leakage']
        values = [true_leakage_amount, detected_leakage_amount, missed_leakage_amount]
        
        bars = plt.bar(categories, values, color=['#ff9999', '#66b3ff', '#ffcc99'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'${height:,.2f}', ha='center', va='bottom')
        
        plt.title('Financial Impact of Revenue Leakage')
        plt.ylabel('Amount ($)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        return plt.gcf()
    
    def generate_report(self):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            dict: Dictionary containing the evaluation report
        """
        metrics = self.calculate_metrics()
        
        # Calculate additional report metrics
        total_records = len(self.ground_truth)
        leakage_records = self.ground_truth['is_leakage'].sum()
        leakage_rate = leakage_records / total_records if total_records > 0 else 0
        
        # Categorize leakage types if available
        if 'leakage_type' in self.ground_truth.columns:
            leakage_by_type = self.ground_truth[self.ground_truth['is_leakage']]['leakage_type'].value_counts().to_dict()
        else:
            leakage_by_type = {}
        
        # Compile report
        report = {
            'metrics': metrics,
            'summary': {
                'total_records': total_records,
                'leakage_records': leakage_records,
                'leakage_rate': leakage_rate,
                'leakage_by_type': leakage_by_type
            }
        }
        
        return report


def validate_detection_system(agent_system, test_data, ground_truth):
    """
    Validate the revenue leakage detection system using test data and ground truth.
    
    Args:
        agent_system: The agent system to validate
        test_data (pd.DataFrame): Test data to run through the system
        ground_truth (pd.DataFrame): Ground truth data for validation
    
    Returns:
        dict: Validation results including metrics and visualizations
    """
    # Run the agent system on test data
    detection_results = agent_system.run(test_data)
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics()
    evaluator.load_data(detection_results, ground_truth)
    
    # Generate evaluation report
    report = evaluator.generate_report()
    
    # Generate visualizations
    confusion_matrix_fig = evaluator.visualize_confusion_matrix()
    financial_impact_fig = evaluator.visualize_financial_impact()
    
    # Compile validation results
    validation_results = {
        'report': report,
        'confusion_matrix': confusion_matrix_fig,
        'financial_impact': financial_impact_fig
    }
    
    return validation_results