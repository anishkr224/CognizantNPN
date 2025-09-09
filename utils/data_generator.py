# Data Generator for AI Revenue Leakage Detection System

import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config
from config import PROCESSED_DATA_DIR

class DataGenerator:
    """Generate synthetic data for AI Revenue Leakage Detection System"""
    
    def __init__(self, num_customers=100, num_invoices=1000, error_rate=0.05):
        """Initialize the data generator
        
        Args:
            num_customers (int): Number of customers to generate
            num_invoices (int): Number of invoices to generate
            error_rate (float): Percentage of records with intentional errors
        """
        self.num_customers = num_customers
        self.num_invoices = num_invoices
        self.error_rate = error_rate
        self.service_types = [
            "cloud_storage", "compute_instances", "database_service", 
            "api_calls", "bandwidth", "support_plan"
        ]
        self.rate_map = {
            "cloud_storage": 0.05,
            "compute_instances": 0.10,
            "database_service": 0.15,
            "api_calls": 0.001,
            "bandwidth": 0.02,
            "support_plan": 50.0
        }
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
        # Create output directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    def generate_contracts(self):
        """Generate synthetic contract data"""
        contracts_data = []
        
        for i in range(1, self.num_customers + 1):
            customer_id = f"C{1000+i}"
            
            # Each customer has contracts for 1-3 service types
            num_services = random.randint(1, 3)
            selected_services = random.sample(self.service_types, num_services)
            
            for service_type in selected_services:
                # Add some variation to the standard rate
                rate_variation = random.uniform(-0.01, 0.01)
                agreed_rate = self.rate_map[service_type] * (1 + rate_variation)
                
                # Contract duration between 6-12 months
                duration_days = random.randint(180, 365)
                start_date = self.start_date + timedelta(days=random.randint(0, 30))
                end_date = start_date + timedelta(days=duration_days)
                
                contracts_data.append({
                    "contract_id": len(contracts_data) + 1,
                    "customer_id": customer_id,
                    "service_type": service_type,
                    "agreed_rate": round(agreed_rate, 4),
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                })
        
        # Save to file
        contracts_file = os.path.join(PROCESSED_DATA_DIR, "contracts.json")
        with open(contracts_file, 'w') as f:
            json.dump(contracts_data, f, indent=2)
            
        print(f"Generated {len(contracts_data)} contracts for {self.num_customers} customers")
        return contracts_data
    
    def generate_billing_records(self, contracts_data):
        """Generate synthetic billing records with intentional errors"""
        billing_data = []
        
        # Create a lookup for contracts by customer and service type
        contract_lookup = {}
        for contract in contracts_data:
            key = (contract["customer_id"], contract["service_type"])
            contract_lookup[key] = contract
        
        for i in range(1, self.num_invoices + 1):
            # Randomly select a contract
            contract = random.choice(contracts_data)
            customer_id = contract["customer_id"]
            service_type = contract["service_type"]
            agreed_rate = contract["agreed_rate"]
            
            # Determine if this record will have an error
            has_error = random.random() < self.error_rate
            error_type = random.choice(["rate", "missing", "duplicate"]) if has_error else None
            
            if error_type == "rate":
                # Incorrect rate error (usually lower than agreed)
                error_direction = -1 if random.random() < 0.8 else 1  # 80% chance of undercharging
                error_magnitude = random.uniform(0.05, 0.2)  # 5-20% error
                billed_rate = agreed_rate * (1 + error_direction * error_magnitude)
                billed_rate = round(billed_rate, 4)
            else:
                billed_rate = agreed_rate
            
            # Generate usage quantity based on service type
            if service_type == "cloud_storage":
                usage_quantity = random.randint(100, 1000)  # GB
            elif service_type == "compute_instances":
                usage_quantity = random.randint(24, 720)  # Hours
            elif service_type == "database_service":
                usage_quantity = random.randint(1, 10)  # Instances
            elif service_type == "api_calls":
                usage_quantity = random.randint(10000, 1000000)  # Calls
            elif service_type == "bandwidth":
                usage_quantity = random.randint(500, 5000)  # GB
            elif service_type == "support_plan":
                usage_quantity = 1  # Flat fee
            
            # Calculate total charge
            total_charge = billed_rate * usage_quantity
            
            # Generate invoice date within contract period
            start_date = datetime.strptime(contract["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(contract["end_date"], "%Y-%m-%d")
            invoice_date = start_date + (end_date - start_date) * random.random()
            
            # Create the billing record
            billing_record = [
                i,  # invoice_id
                customer_id,
                service_type,
                billed_rate,
                usage_quantity,
                round(total_charge, 2),
                invoice_date.strftime("%Y-%m-%d")
            ]
            
            billing_data.append(billing_record)
            
            # If this is a duplicate error, add the same record again with a new invoice ID
            if error_type == "duplicate":
                duplicate_record = billing_record.copy()
                duplicate_record[0] = len(billing_data) + 1  # New invoice ID
                billing_data.append(duplicate_record)
        
        # For missing charge errors, we'll remove some records for services that should be billed
        if "missing" in [error_type for error_type in [random.choice(["rate", "missing", "duplicate"]) 
                                                 for _ in range(int(self.num_invoices * self.error_rate))] 
                         if error_type is not None]:
            # We've already generated all records, so we don't need to remove any
            pass
        
        # Convert to DataFrame and save
        columns = ['invoice_id', 'customer_id', 'service_type', 'billed_rate', 
                  'usage_quantity', 'total_charge', 'date']
        billing_df = pd.DataFrame(billing_data, columns=columns)
        
        # Save to file
        billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
        billing_df.to_csv(billing_file, index=False)
        
        print(f"Generated {len(billing_data)} billing records with {self.error_rate*100}% error rate")
        return billing_df
    
    def generate_usage_logs(self, contracts_data):
        """Generate synthetic usage logs"""
        usage_logs = []
        
        for i in range(1, self.num_invoices * 3):  # More granular than invoices
            # Randomly select a contract
            contract = random.choice(contracts_data)
            customer_id = contract["customer_id"]
            service_type = contract["service_type"]
            
            # Generate usage data
            if service_type == "cloud_storage":
                recorded_usage = random.randint(10, 100)  # GB (daily usage)
            elif service_type == "compute_instances":
                recorded_usage = random.randint(1, 24)  # Hours (daily usage)
            elif service_type == "database_service":
                recorded_usage = 1  # Instance (constant)
            elif service_type == "api_calls":
                recorded_usage = random.randint(1000, 100000)  # Calls (daily)
            elif service_type == "bandwidth":
                recorded_usage = random.randint(50, 500)  # GB (daily)
            elif service_type == "support_plan":
                recorded_usage = 1  # Flat fee
            
            # Generate timestamp within contract period
            start_date = datetime.strptime(contract["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(contract["end_date"], "%Y-%m-%d")
            timestamp = start_date + (end_date - start_date) * random.random()
            
            usage_logs.append({
                "log_id": i,
                "customer_id": customer_id,
                "service_type": service_type,
                "recorded_usage": recorded_usage,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Save to file
        usage_file = os.path.join(PROCESSED_DATA_DIR, "usage_logs.json")
        with open(usage_file, 'w') as f:
            json.dump(usage_logs, f, indent=2)
            
        print(f"Generated {len(usage_logs)} usage logs")
        return usage_logs
    
    def generate_service_provisioning(self, contracts_data):
        """Generate synthetic service provisioning records"""
        provisioning_data = []
        
        for i, contract in enumerate(contracts_data):
            customer_id = contract["customer_id"]
            service_type = contract["service_type"]
            
            # Determine provisioned level based on service type
            if service_type == "cloud_storage":
                provisioned_level = f"{random.randint(1, 10)}TB"
            elif service_type == "compute_instances":
                instance_types = ["small", "medium", "large", "xlarge"]
                provisioned_level = f"{random.choice(instance_types)}"
            elif service_type == "database_service":
                db_types = ["standard", "high-memory", "high-cpu", "enterprise"]
                provisioned_level = f"{random.choice(db_types)}"
            elif service_type == "api_calls":
                provisioned_level = f"{random.choice(['basic', 'standard', 'premium'])}"
            elif service_type == "bandwidth":
                provisioned_level = f"{random.randint(1, 10)}Gbps"
            elif service_type == "support_plan":
                provisioned_level = f"{random.choice(['basic', 'standard', 'premium', 'enterprise'])}"
            
            # Status is usually active, but sometimes pending or suspended
            status_options = ["active", "active", "active", "active", "pending", "suspended"]
            status = random.choice(status_options)
            
            provisioning_data.append([
                i + 1,  # provision_id
                customer_id,
                service_type,
                provisioned_level,
                status
            ])
        
        # Convert to DataFrame and save
        columns = ['provision_id', 'customer_id', 'service_type', 'provisioned_level', 'status']
        provisioning_df = pd.DataFrame(provisioning_data, columns=columns)
        
        # Save to file
        provisioning_file = os.path.join(PROCESSED_DATA_DIR, "service_provisioning.csv")
        provisioning_df.to_csv(provisioning_file, index=False)
        
        print(f"Generated {len(provisioning_data)} service provisioning records")
        return provisioning_df
    
    def generate_all_data(self):
        """Generate all synthetic datasets"""
        contracts_data = self.generate_contracts()
        billing_records = self.generate_billing_records(contracts_data)
        usage_logs = self.generate_usage_logs(contracts_data)
        service_provisioning = self.generate_service_provisioning(contracts_data)
        
        return {
            "contracts": contracts_data,
            "billing_records": billing_records,
            "usage_logs": usage_logs,
            "service_provisioning": service_provisioning
        }


if __name__ == "__main__":
    # Generate synthetic data
    generator = DataGenerator(num_customers=100, num_invoices=1000, error_rate=0.05)
    data = generator.generate_all_data()
    
    print("\nData generation complete. Files saved to:", PROCESSED_DATA_DIR)