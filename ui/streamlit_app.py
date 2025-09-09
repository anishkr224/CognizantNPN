# Streamlit UI for AI Revenue Leakage Detection System

import os
import sys
import json
import pandas as pd
import streamlit as st
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config
from config import APP_TITLE, APP_DESCRIPTION, PROCESSED_DATA_DIR

# Import agent system
from agents.agents import AgentSystem

# Import data generator
from utils.data_generator import DataGenerator

# Import knowledge base
from utils.knowledge_base import KnowledgeBase


def save_uploaded_file(uploaded_file, destination):
    """Save an uploaded file to the specified destination"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return destination


def main():
    """Main Streamlit application"""
    st.set_page_config(page_title=APP_TITLE, page_icon="💰", layout="wide")
    
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Data Upload", "Generate Sample Data", "Run Audit", "View Results"])
    
    if page == "Home":
        show_home_page()
    elif page == "Data Upload":
        show_data_upload_page()
    elif page == "Generate Sample Data":
        show_generate_sample_data_page()
    elif page == "Run Audit":
        show_run_audit_page()
    elif page == "View Results":
        show_view_results_page()


def show_home_page():
    """Show the home page"""
    st.header("Welcome to the AI Revenue Leakage Detection System")
    
    st.markdown("""
    This system helps you identify and prioritize revenue leakage within your billing workflows by analyzing:
    
    - **Billing Records**: Your invoices and charges
    - **Contracts**: Your customer agreements and rates
    - **Usage Logs**: Actual service usage data
    - **Service Provisioning**: What services are set up for each customer
    
    ### Key Features
    
    - **Detect Discrepancies**: Find missing charges, incorrect rates, usage mismatches, and duplicate entries
    - **Near Real-Time Alerts**: Get notified of issues as they arise
    - **Root Cause Analysis**: Understand why discrepancies occur
    - **Revenue Recovery**: Accelerate the recovery of lost revenue
    
    ### Getting Started
    
    1. Upload your data files or generate sample data
    2. Run the audit to analyze your data
    3. View the results and take action
    """)
    
    # Check if data exists
    contracts_file = os.path.join(PROCESSED_DATA_DIR, "contracts.json")
    billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
    usage_file = os.path.join(PROCESSED_DATA_DIR, "usage_logs.json")
    provisioning_file = os.path.join(PROCESSED_DATA_DIR, "service_provisioning.csv")
    
    data_exists = all(os.path.exists(f) for f in [contracts_file, billing_file, usage_file, provisioning_file])
    
    if data_exists:
        st.success("✅ Data files are ready for analysis")
        st.button("Run Audit", on_click=lambda: st.session_state.update({"page": "Run Audit"}))
    else:
        st.warning("⚠️ No data files found. Please upload data or generate sample data.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Upload Data", on_click=lambda: st.session_state.update({"page": "Data Upload"}))
        with col2:
            st.button("Generate Sample Data", on_click=lambda: st.session_state.update({"page": "Generate Sample Data"}))


def show_data_upload_page():
    """Show the data upload page"""
    st.header("Upload Your Data")
    
    st.markdown("""
    Upload your data files to analyze for revenue leakage. The system requires the following files:
    
    - **Contracts**: JSON file with contract details (customer ID, service type, agreed rate, etc.)
    - **Billing Records**: CSV file with billing details (invoice ID, customer ID, service type, billed rate, etc.)
    - **Usage Logs**: JSON file with usage details (customer ID, service type, recorded usage, timestamp)
    - **Service Provisioning**: CSV file with provisioning details (customer ID, service type, provisioned level, status)
    """)
    
    # Create columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contracts (JSON)")
        contracts_file = st.file_uploader("Upload contracts file", type=["json"])
        
        st.subheader("Usage Logs (JSON)")
        usage_file = st.file_uploader("Upload usage logs file", type=["json"])
    
    with col2:
        st.subheader("Billing Records (CSV)")
        billing_file = st.file_uploader("Upload billing records file", type=["csv"])
        
        st.subheader("Service Provisioning (CSV)")
        provisioning_file = st.file_uploader("Upload service provisioning file", type=["csv"])
    
    if st.button("Upload Files"):
        if not all([contracts_file, billing_file, usage_file, provisioning_file]):
            st.error("Please upload all required files.")
            return
        
        # Create processed data directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Save uploaded files
        contracts_path = save_uploaded_file(contracts_file, os.path.join(PROCESSED_DATA_DIR, "contracts.json"))
        billing_path = save_uploaded_file(billing_file, os.path.join(PROCESSED_DATA_DIR, "billing_records.csv"))
        usage_path = save_uploaded_file(usage_file, os.path.join(PROCESSED_DATA_DIR, "usage_logs.json"))
        provisioning_path = save_uploaded_file(provisioning_file, os.path.join(PROCESSED_DATA_DIR, "service_provisioning.csv"))
        
        st.success("Files uploaded successfully!")
        
        # Create vector store
        with st.spinner("Creating knowledge base..."):
            kb = KnowledgeBase()
            kb.create_vector_store()
            st.success("Knowledge base created successfully!")
        
        # Show run audit button
        st.button("Run Audit", on_click=lambda: st.session_state.update({"page": "Run Audit"}))


def show_generate_sample_data_page():
    """Show the generate sample data page"""
    st.header("Generate Sample Data")
    
    st.markdown("""
    Generate synthetic data for testing the system. This will create sample contracts, billing records, usage logs, and service provisioning records with intentional errors for demonstration purposes.
    """)
    
    # Data generation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_customers = st.number_input("Number of Customers", min_value=10, max_value=1000, value=100, step=10)
    
    with col2:
        num_invoices = st.number_input("Number of Invoices", min_value=100, max_value=10000, value=1000, step=100)
    
    with col3:
        error_rate = st.slider("Error Rate (%)", min_value=1, max_value=20, value=5, step=1) / 100
    
    if st.button("Generate Data"):
        with st.spinner("Generating sample data..."):
            # Create data generator
            generator = DataGenerator(num_customers=num_customers, num_invoices=num_invoices, error_rate=error_rate)
            
            # Generate data
            data = generator.generate_all_data()
            
            # Create vector store
            kb = KnowledgeBase()
            kb.create_vector_store()
            
            st.success(f"Sample data generated successfully! Generated {len(data['contracts'])} contracts, {len(data['billing_records'])} billing records, {len(data['usage_logs'])} usage logs, and {len(data['service_provisioning'])} service provisioning records.")
        
        # Show run audit button
        st.button("Run Audit", on_click=lambda: st.session_state.update({"page": "Run Audit"}))


def show_run_audit_page():
    """Show the run audit page"""
    st.header("Run Revenue Leakage Audit")
    
    # Check if data exists
    contracts_file = os.path.join(PROCESSED_DATA_DIR, "contracts.json")
    billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
    usage_file = os.path.join(PROCESSED_DATA_DIR, "usage_logs.json")
    provisioning_file = os.path.join(PROCESSED_DATA_DIR, "service_provisioning.csv")
    
    data_exists = all(os.path.exists(f) for f in [contracts_file, billing_file, usage_file, provisioning_file])
    
    if not data_exists:
        st.error("No data files found. Please upload data or generate sample data.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Upload Data", on_click=lambda: st.session_state.update({"page": "Data Upload"}))
        with col2:
            st.button("Generate Sample Data", on_click=lambda: st.session_state.update({"page": "Generate Sample Data"}))
        return
    
    st.markdown("""
    Run the AI-powered audit to detect revenue leakage in your data. The system will analyze your contracts, billing records, usage logs, and service provisioning data to identify discrepancies.
    
    The audit will detect:
    - **Incorrect Rates**: Billed rates that don't match contracted rates
    - **Missing Charges**: Services that should be billed but aren't
    - **Duplicate Entries**: Multiple charges for the same service
    - **Usage Mismatches**: Discrepancies between recorded usage and billed usage
    """)
    
    # Show data summary
    st.subheader("Data Summary")
    
    try:
        with open(contracts_file, 'r') as f:
            contracts = json.load(f)
        
        billing_df = pd.read_csv(billing_file)
        
        with open(usage_file, 'r') as f:
            usage_logs = json.load(f)
        
        provisioning_df = pd.read_csv(provisioning_file)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Contracts", len(contracts))
        
        with col2:
            st.metric("Billing Records", len(billing_df))
        
        with col3:
            st.metric("Usage Logs", len(usage_logs))
        
        with col4:
            st.metric("Provisioning Records", len(provisioning_df))
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Run audit button
    if st.button("Run Audit"):
        with st.spinner("Running audit... This may take a few minutes."):
            try:
                # Create agent system
                agent_system = AgentSystem()
                
                # Run audit
                report = agent_system.run_audit()
                
                # Save report
                report_file = os.path.join(PROCESSED_DATA_DIR, f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                with open(report_file, 'w') as f:
                    f.write(report)
                
                st.session_state["latest_report"] = report
                st.session_state["latest_report_file"] = report_file
                
                st.success("Audit completed successfully!")
                st.button("View Results", on_click=lambda: st.session_state.update({"page": "View Results"}))
            
            except Exception as e:
                error_message = f"Error running audit: {str(e)}"
                st.error(error_message)
                print("\n" + "="*50)
                print("ERROR DETAILS:")
                print(error_message)
                print("="*50 + "\n")


def show_view_results_page():
    """Show the view results page"""
    st.header("Audit Results")
    
    # Check if there's a report in session state
    if "latest_report" in st.session_state:
        report = st.session_state["latest_report"]
        st.markdown(report)
        
        # Download button
        if "latest_report_file" in st.session_state:
            with open(st.session_state["latest_report_file"], "r") as f:
                report_content = f.read()
            
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"revenue_leakage_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    else:
        # Check if there are any reports in the processed data directory
        report_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.startswith("audit_report_") and f.endswith(".md")]
        
        if report_files:
            # Sort by date (newest first)
            report_files.sort(reverse=True)
            
            # Let user select a report
            selected_report = st.selectbox("Select a report", report_files)
            
            # Display the selected report
            report_path = os.path.join(PROCESSED_DATA_DIR, selected_report)
            with open(report_path, "r") as f:
                report_content = f.read()
            
            st.markdown(report_content)
            
            # Download button
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"revenue_leakage_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("No audit reports found. Please run an audit first.")
            st.button("Run Audit", on_click=lambda: st.session_state.update({"page": "Run Audit"}))


if __name__ == "__main__":
    main()