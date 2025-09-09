# Tools for AI Revenue Leakage Detection System Agents

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utils
from utils.knowledge_base import KnowledgeBase
from config import PROCESSED_DATA_DIR


def retrieve_contract_info(query: str) -> str:
    """Retrieve contract information using RAG
    
    Args:
        query: The query to search for in the knowledge base
        
    Returns:
        The retrieved contract information
    """
    kb = KnowledgeBase()
    docs = kb.similarity_search(query)
    return docs[0].page_content if docs else "No relevant contract information found."


def query_billing_data(query: str) -> str:
    """Query billing records
    
    Args:
        query: A description of the billing data to query
        
    Returns:
        The queried billing data as a string
    """
    billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
    if not os.path.exists(billing_file):
        return "Billing records file not found."
    
    billing_df = pd.read_csv(billing_file)
    
    # Simple keyword-based filtering
    if "customer" in query.lower():
        customer_id = None
        for word in query.split():
            if word.startswith("C") and len(word) >= 5:
                customer_id = word
                break
        
        if customer_id:
            filtered_df = billing_df[billing_df['customer_id'] == customer_id]
            return filtered_df.to_string() if not filtered_df.empty else f"No billing records found for {customer_id}"
    
    if "service" in query.lower():
        service_type = None
        service_types = ["cloud_storage", "compute_instances", "database_service", 
                        "api_calls", "bandwidth", "support_plan"]
        
        for service in service_types:
            if service in query.lower():
                service_type = service
                break
        
        if service_type:
            filtered_df = billing_df[billing_df['service_type'] == service_type]
            return filtered_df.to_string() if not filtered_df.empty else f"No billing records found for {service_type}"
    
    # Default: return all billing data
    return billing_df.to_string()


def compare_rates(customer_id: str = None, service_type: str = None) -> str:
    """Compare contracted rates with billed rates
    
    Args:
        customer_id: The customer ID to compare rates for
        service_type: The service type to compare rates for
        
    Returns:
        A report of rate discrepancies
    """
    # Load contracts
    contracts_file = os.path.join(PROCESSED_DATA_DIR, "contracts.json")
    if not os.path.exists(contracts_file):
        return "Contracts file not found."
    
    with open(contracts_file, 'r') as f:
        contracts = json.load(f)
    
    # Load billing records
    billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
    if not os.path.exists(billing_file):
        return "Billing records file not found."
    
    billing_df = pd.read_csv(billing_file)
    
    # Filter data if needed
    if customer_id:
        contracts = [c for c in contracts if c['customer_id'] == customer_id]
        billing_df = billing_df[billing_df['customer_id'] == customer_id]
    
    if service_type:
        contracts = [c for c in contracts if c['service_type'] == service_type]
        billing_df = billing_df[billing_df['service_type'] == service_type]
    
    # Create a lookup for contracts by customer and service type
    contract_lookup = {}
    for contract in contracts:
        key = (contract["customer_id"], contract["service_type"])
        contract_lookup[key] = contract
    
    # Compare rates and find discrepancies
    discrepancies = []
    for _, row in billing_df.iterrows():
        key = (row['customer_id'], row['service_type'])
        if key in contract_lookup:
            contract = contract_lookup[key]
            agreed_rate = contract['agreed_rate']
            billed_rate = row['billed_rate']
            
            # Check if rates match (with small tolerance for floating point comparison)
            if abs(agreed_rate - billed_rate) > 0.0001:
                discrepancy = {
                    'invoice_id': row['invoice_id'],
                    'customer_id': row['customer_id'],
                    'service_type': row['service_type'],
                    'agreed_rate': agreed_rate,
                    'billed_rate': billed_rate,
                    'usage_quantity': row['usage_quantity'],
                    'total_charge': row['total_charge'],
                    'correct_charge': agreed_rate * row['usage_quantity'],
                    'revenue_impact': (agreed_rate - billed_rate) * row['usage_quantity'],
                    'date': row['date']
                }
                discrepancies.append(discrepancy)
    
    # Generate report
    if not discrepancies:
        return "No rate discrepancies found."
    
    report = f"Found {len(discrepancies)} rate discrepancies:\n\n"
    
    # Calculate total revenue impact
    total_impact = sum(d['revenue_impact'] for d in discrepancies)
    report += f"Total Revenue Impact: ${total_impact:.2f}\n\n"
    
    # Add details for each discrepancy
    for i, d in enumerate(discrepancies[:10]):  # Limit to first 10 for readability
        report += f"Discrepancy {i+1}:\n"
        report += f"  Invoice ID: {d['invoice_id']}\n"
        report += f"  Customer: {d['customer_id']}\n"
        report += f"  Service: {d['service_type']}\n"
        report += f"  Agreed Rate: ${d['agreed_rate']}\n"
        report += f"  Billed Rate: ${d['billed_rate']}\n"
        report += f"  Usage: {d['usage_quantity']}\n"
        report += f"  Billed Amount: ${d['total_charge']:.2f}\n"
        report += f"  Correct Amount: ${d['correct_charge']:.2f}\n"
        report += f"  Revenue Impact: ${d['revenue_impact']:.2f}\n"
        report += f"  Date: {d['date']}\n\n"
    
    if len(discrepancies) > 10:
        report += f"... and {len(discrepancies) - 10} more discrepancies.\n"
    
    return report


def detect_missing_charges() -> str:
    """Detect missing charges by comparing contracts with billing records
    
    Returns:
        A report of missing charges
    """
    # Load contracts
    contracts_file = os.path.join(PROCESSED_DATA_DIR, "contracts.json")
    if not os.path.exists(contracts_file):
        return "Contracts file not found."
    
    with open(contracts_file, 'r') as f:
        contracts = json.load(f)
    
    # Load billing records
    billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
    if not os.path.exists(billing_file):
        return "Billing records file not found."
    
    billing_df = pd.read_csv(billing_file)
    
    # Create a set of (customer_id, service_type) pairs from billing records
    billed_services = set()
    for _, row in billing_df.iterrows():
        billed_services.add((row['customer_id'], row['service_type']))
    
    # Find contracts without corresponding billing records
    missing_charges = []
    for contract in contracts:
        key = (contract['customer_id'], contract['service_type'])
        if key not in billed_services:
            missing_charges.append(contract)
    
    # Generate report
    if not missing_charges:
        return "No missing charges found."
    
    report = f"Found {len(missing_charges)} missing charges:\n\n"
    
    # Add details for each missing charge
    for i, contract in enumerate(missing_charges[:10]):  # Limit to first 10 for readability
        report += f"Missing Charge {i+1}:\n"
        report += f"  Contract ID: {contract['contract_id']}\n"
        report += f"  Customer: {contract['customer_id']}\n"
        report += f"  Service: {contract['service_type']}\n"
        report += f"  Agreed Rate: ${contract['agreed_rate']}\n"
        report += f"  Contract Period: {contract['start_date']} to {contract['end_date']}\n\n"
    
    if len(missing_charges) > 10:
        report += f"... and {len(missing_charges) - 10} more missing charges.\n"
    
    return report


def detect_duplicate_entries() -> str:
    """Detect duplicate billing entries
    
    Returns:
        A report of duplicate entries
    """
    # Load billing records
    billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
    if not os.path.exists(billing_file):
        return "Billing records file not found."
    
    billing_df = pd.read_csv(billing_file)
    
    # Group by all columns except invoice_id to find duplicates
    duplicate_columns = ['customer_id', 'service_type', 'billed_rate', 'usage_quantity', 'total_charge', 'date']
    duplicates = billing_df[billing_df.duplicated(subset=duplicate_columns, keep=False)]
    
    # Sort by duplicate columns to group duplicates together
    duplicates = duplicates.sort_values(by=duplicate_columns)
    
    # Generate report
    if duplicates.empty:
        return "No duplicate entries found."
    
    report = f"Found {len(duplicates)} duplicate entries:\n\n"
    
    # Add details for each duplicate group
    duplicate_groups = duplicates.groupby(duplicate_columns)
    for i, (group_key, group_df) in enumerate(duplicate_groups):
        if i >= 10:  # Limit to first 10 groups for readability
            break
        
        customer_id, service_type, billed_rate, usage_quantity, total_charge, date = group_key
        invoice_ids = group_df['invoice_id'].tolist()
        
        report += f"Duplicate Group {i+1}:\n"
        report += f"  Customer: {customer_id}\n"
        report += f"  Service: {service_type}\n"
        report += f"  Billed Rate: ${billed_rate}\n"
        report += f"  Usage: {usage_quantity}\n"
        report += f"  Total Charge: ${total_charge}\n"
        report += f"  Date: {date}\n"
        report += f"  Invoice IDs: {', '.join(map(str, invoice_ids))}\n\n"
    
    if len(duplicate_groups) > 10:
        report += f"... and {len(duplicate_groups) - 10} more duplicate groups.\n"
    
    return report


def detect_usage_mismatches() -> str:
    """Detect mismatches between usage logs and billing records
    
    Returns:
        A report of usage mismatches
    """
    # Load usage logs
    usage_file = os.path.join(PROCESSED_DATA_DIR, "usage_logs.json")
    if not os.path.exists(usage_file):
        return "Usage logs file not found."
    
    with open(usage_file, 'r') as f:
        usage_logs = json.load(f)
    
    # Load billing records
    billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
    if not os.path.exists(billing_file):
        return "Billing records file not found."
    
    billing_df = pd.read_csv(billing_file)
    
    # Aggregate usage by customer and service type
    usage_by_customer_service = {}
    for log in usage_logs:
        key = (log['customer_id'], log['service_type'])
        if key not in usage_by_customer_service:
            usage_by_customer_service[key] = 0
        usage_by_customer_service[key] += log['recorded_usage']
    
    # Aggregate billing by customer and service type
    billing_by_customer_service = {}
    for _, row in billing_df.iterrows():
        key = (row['customer_id'], row['service_type'])
        if key not in billing_by_customer_service:
            billing_by_customer_service[key] = 0
        billing_by_customer_service[key] += row['usage_quantity']
    
    # Find mismatches
    mismatches = []
    for key in set(usage_by_customer_service.keys()) | set(billing_by_customer_service.keys()):
        usage = usage_by_customer_service.get(key, 0)
        billed = billing_by_customer_service.get(key, 0)
        
        # Check if there's a significant mismatch (more than 10% difference)
        if usage > 0 and billed > 0:
            difference_pct = abs(usage - billed) / max(usage, billed) * 100
            if difference_pct > 10:
                customer_id, service_type = key
                mismatches.append({
                    'customer_id': customer_id,
                    'service_type': service_type,
                    'recorded_usage': usage,
                    'billed_usage': billed,
                    'difference': usage - billed,
                    'difference_pct': difference_pct
                })
    
    # Generate report
    if not mismatches:
        return "No significant usage mismatches found."
    
    report = f"Found {len(mismatches)} significant usage mismatches:\n\n"
    
    # Add details for each mismatch
    for i, mismatch in enumerate(mismatches[:10]):  # Limit to first 10 for readability
        report += f"Mismatch {i+1}:\n"
        report += f"  Customer: {mismatch['customer_id']}\n"
        report += f"  Service: {mismatch['service_type']}\n"
        report += f"  Recorded Usage: {mismatch['recorded_usage']}\n"
        report += f"  Billed Usage: {mismatch['billed_usage']}\n"
        report += f"  Difference: {mismatch['difference']} ({mismatch['difference_pct']:.2f}%)\n\n"
    
    if len(mismatches) > 10:
        report += f"... and {len(mismatches) - 10} more mismatches.\n"
    
    return report