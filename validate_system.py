# Validation Script for AI Revenue Leakage Detection System

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.data_generator import DataGenerator
from utils.knowledge_base import KnowledgeBase
from agents.agents import AgentSystem
from utils.evaluation import EvaluationMetrics, validate_detection_system
import config

def generate_validation_data():
    """
    Generate synthetic data for validation with known ground truth.
    
    Returns:
        tuple: (test_data, ground_truth) DataFrames
    """
    # Initialize data generator
    data_gen = DataGenerator()
    
    # Generate synthetic data with known errors
    data_gen.generate_data(num_contracts=20, num_bills=50, error_rate=0.2)
    
    # Load the generated data
    billing_data = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'billing_data.csv'))
    contract_data = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'contract_data.csv'))
    usage_data = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'usage_logs.csv'))
    service_data = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'service_provisioning.csv'))
    
    # Create ground truth by identifying the intentional errors
    ground_truth = []
    
    # Check for missing charges
    for _, contract in contract_data.iterrows():
        contract_id = contract['contract_id']
        service_ids = contract['service_ids'].split(',')
        
        for service_id in service_ids:
            service_id = service_id.strip()
            # Check if this service has a corresponding billing entry
            billed = billing_data[
                (billing_data['contract_id'] == contract_id) & 
                (billing_data['service_id'] == service_id)
            ].shape[0] > 0
            
            if not billed:
                ground_truth.append({
                    'contract_id': contract_id,
                    'service_id': service_id,
                    'is_leakage': True,
                    'leakage_type': 'missing_charge',
                    'amount': float(contract[contract['service_ids'].find(service_id) + len(service_id) + 1:].split(',')[0].strip())
                })
    
    # Check for incorrect rates
    for _, bill in billing_data.iterrows():
        contract_id = bill['contract_id']
        service_id = bill['service_id']
        billed_amount = bill['amount']
        
        # Find the contract rate for this service
        contract_row = contract_data[contract_data['contract_id'] == contract_id]
        if not contract_row.empty:
            contract_info = contract_row.iloc[0]
            service_index = contract_info['service_ids'].split(',').index(service_id) if service_id in contract_info['service_ids'] else -1
            
            if service_index >= 0:
                contract_rate = float(contract_info['service_rates'].split(',')[service_index].strip())
                
                # Check if billed amount matches contract rate
                if abs(billed_amount - contract_rate) > 0.01:  # Allow for small floating point differences
                    ground_truth.append({
                        'contract_id': contract_id,
                        'service_id': service_id,
                        'is_leakage': True,
                        'leakage_type': 'incorrect_rate',
                        'amount': abs(contract_rate - billed_amount)
                    })
    
    # Check for usage mismatches
    for _, bill in billing_data.iterrows():
        contract_id = bill['contract_id']
        service_id = bill['service_id']
        
        # Find usage for this service
        usage_row = usage_data[
            (usage_data['contract_id'] == contract_id) & 
            (usage_data['service_id'] == service_id)
        ]
        
        if not usage_row.empty:
            usage = usage_row.iloc[0]['usage_amount']
            
            # Find the contract usage rate
            contract_row = contract_data[contract_data['contract_id'] == contract_id]
            if not contract_row.empty:
                contract_info = contract_row.iloc[0]
                service_index = contract_info['service_ids'].split(',').index(service_id) if service_id in contract_info['service_ids'] else -1
                
                if service_index >= 0:
                    contract_rate = float(contract_info['service_rates'].split(',')[service_index].strip())
                    expected_amount = usage * contract_rate
                    
                    # Check if billed amount matches expected amount based on usage
                    if abs(bill['amount'] - expected_amount) > 0.01:  # Allow for small floating point differences
                        ground_truth.append({
                            'contract_id': contract_id,
                            'service_id': service_id,
                            'is_leakage': True,
                            'leakage_type': 'usage_mismatch',
                            'amount': abs(expected_amount - bill['amount'])
                        })
    
    # Check for duplicate entries
    for contract_id in billing_data['contract_id'].unique():
        for service_id in billing_data[billing_data['contract_id'] == contract_id]['service_id'].unique():
            count = billing_data[
                (billing_data['contract_id'] == contract_id) & 
                (billing_data['service_id'] == service_id)
            ].shape[0]
            
            if count > 1:
                # Get the duplicate entries
                duplicates = billing_data[
                    (billing_data['contract_id'] == contract_id) & 
                    (billing_data['service_id'] == service_id)
                ]
                
                # Add each duplicate as a separate leakage instance
                for i in range(1, len(duplicates)):
                    ground_truth.append({
                        'contract_id': contract_id,
                        'service_id': service_id,
                        'is_leakage': True,
                        'leakage_type': 'duplicate_entry',
                        'amount': duplicates.iloc[i]['amount']
                    })
    
    # Add non-leakage entries for completeness
    for _, bill in billing_data.iterrows():
        # Check if this bill is already in ground truth as a leakage
        is_leakage = any(
            (g['contract_id'] == bill['contract_id'] and 
             g['service_id'] == bill['service_id'] and 
             g['is_leakage']) 
            for g in ground_truth
        )
        
        if not is_leakage:
            ground_truth.append({
                'contract_id': bill['contract_id'],
                'service_id': bill['service_id'],
                'is_leakage': False,
                'leakage_type': 'none',
                'amount': 0.0
            })
    
    # Convert to DataFrame
    ground_truth_df = pd.DataFrame(ground_truth)
    
    # Combine all data for testing
    test_data = {
        'billing_data': billing_data,
        'contract_data': contract_data,
        'usage_data': usage_data,
        'service_data': service_data
    }
    
    return test_data, ground_truth_df

def run_validation():
    """
    Run the validation process and generate a report.
    """
    print("Generating validation data...")
    test_data, ground_truth = generate_validation_data()
    
    print("Initializing knowledge base...")
    kb = KnowledgeBase()
    kb.load_data()
    
    print("Setting up agent system...")
    agent_system = AgentSystem(kb)
    
    print("Running validation...")
    # Convert test_data dict to the format expected by the agent system
    detection_results = agent_system.run(test_data)
    
    # Format detection results to match ground truth structure
    formatted_results = []
    for result in detection_results:
        formatted_results.append({
            'contract_id': result['contract_id'],
            'service_id': result['service_id'],
            'is_detected': True,
            'detected_type': result['issue_type'],
            'detected_amount': result['amount']
        })
    
    # Add non-detected entries
    for _, bill in test_data['billing_data'].iterrows():
        # Check if this bill is already in formatted_results
        is_detected = any(
            (r['contract_id'] == bill['contract_id'] and 
             r['service_id'] == bill['service_id']) 
            for r in formatted_results
        )
        
        if not is_detected:
            formatted_results.append({
                'contract_id': bill['contract_id'],
                'service_id': bill['service_id'],
                'is_detected': False,
                'detected_type': 'none',
                'detected_amount': 0.0
            })
    
    results_df = pd.DataFrame(formatted_results)
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics()
    evaluator.load_data(results_df, ground_truth)
    
    # Generate evaluation report
    report = evaluator.generate_report()
    
    # Generate visualizations
    confusion_matrix_fig = evaluator.visualize_confusion_matrix()
    financial_impact_fig = evaluator.visualize_financial_impact()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save report as JSON
    with open(os.path.join(results_dir, f'validation_report_{timestamp}.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        report_serializable = json.loads(
            json.dumps(report, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        )
        json.dump(report_serializable, f, indent=4)
    
    # Save visualizations
    confusion_matrix_fig.savefig(os.path.join(results_dir, f'confusion_matrix_{timestamp}.png'))
    financial_impact_fig.savefig(os.path.join(results_dir, f'financial_impact_{timestamp}.png'))
    
    print(f"Validation complete. Results saved to {results_dir}")
    
    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"Precision: {report['metrics']['precision']:.2f}")
    print(f"Recall: {report['metrics']['recall']:.2f}")
    print(f"F1 Score: {report['metrics']['f1_score']:.2f}")
    print(f"Recovery Rate: {report['metrics']['recovery_rate']:.2f}")
    print(f"True Leakage Amount: ${report['metrics']['true_leakage_amount']:.2f}")
    print(f"Detected Leakage Amount: ${report['metrics']['detected_leakage_amount']:.2f}")
    
    return report, results_dir

if __name__ == "__main__":
    run_validation()