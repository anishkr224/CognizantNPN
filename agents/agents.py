# Agent Definitions for AI Revenue Leakage Detection System

import os
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config
from config import LLM_MODEL, LLM_TEMPERATURE

# Import tools
from agents.tools import (
    retrieve_contract_info,
    query_billing_data,
    compare_rates,
    detect_missing_charges,
    detect_duplicate_entries,
    detect_usage_mismatches
)

class AgentSystem:
    """Agent System for AI Revenue Leakage Detection"""
    
    def __init__(self):
        """Initialize the agent system"""
        try:
            from crewai import Agent, Task, Crew
            from langchain.tools import Tool
            from langchain_google_genai import ChatGoogleGenerativeAI
            from config import GEMINI_API_KEY
            import google.generativeai as genai
            
            # Configure Google Generative AI with API key
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Initialize LLM with explicit API key
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL, 
                temperature=LLM_TEMPERATURE,
                google_api_key=GEMINI_API_KEY
            )
            
            # Define tools directly in the agent definition
            
            # Define agents
            self.data_ingestion_agent = Agent(
                role="Data Quality Specialist",
                goal="Ensure all required data (billing, contracts, usage) is available and clean.",
                backstory="Expert in data pipelines and ETL processes.",
                tools=[],
                verbose=True,
                llm=self.llm
            )
            
            self.analysis_agent = Agent(
                role="Forensic Billing Auditor",
                goal="Identify discrepancies between contracted rates and billed amounts.",
                backstory="A meticulous auditor with years of experience in finding financial errors.",
                tools=[
                    Tool.from_function(
                        func=retrieve_contract_info,
                        name="RetrieveContractTerms",
                        description="Useful for retrieving agreed rates and terms from customer contracts."
                    ),
                    Tool.from_function(
                        func=query_billing_data,
                        name="QueryBillingRecords",
                        description="Useful for querying billing records to find invoices and charges."
                    ),
                    Tool.from_function(
                        func=compare_rates,
                        name="CompareRates",
                        description="Compare contracted rates with billed rates to find discrepancies."
                    ),
                    Tool.from_function(
                        func=detect_missing_charges,
                        name="DetectMissingCharges",
                        description="Detect missing charges by comparing contracts with billing records."
                    ),
                    Tool.from_function(
                        func=detect_duplicate_entries,
                        name="DetectDuplicateEntries",
                        description="Detect duplicate billing entries."
                    ),
                    Tool.from_function(
                        func=detect_usage_mismatches,
                        name="DetectUsageMismatches",
                        description="Detect mismatches between usage logs and billing records."
                    )
                ],
                verbose=True,
                llm=self.llm
            )
            
            self.reporting_agent = Agent(
                role="Compliance Reporting Officer",
                goal="Generate clear and concise reports on found discrepancies and recommend actions.",
                backstory="Skilled in communicating complex financial issues to stakeholders.",
                tools=[],
                verbose=True,
                llm=self.llm
            )
            
            # Define tasks
            self.ingestion_task = Task(
                description="Load and clean the data from billing records, contracts, usage logs, and service provisioning records.",
                agent=self.data_ingestion_agent,
                expected_output="Cleaned datasets ready for analysis."
            )
            
            self.analysis_task = Task(
                description="Analyze the data to find discrepancies: incorrect rates, missing charges, duplicate entries, and usage mismatches.",
                agent=self.analysis_agent,
                expected_output="A comprehensive list of all detected discrepancies with details."
            )
            
            self.reporting_task = Task(
                description="Summarize the findings from the analysis task. Create a report for management highlighting the total number of errors, estimated revenue loss, and recommended actions.",
                agent=self.reporting_agent,
                expected_output="A well-structured report in markdown format."
            )
            
            # Form crew
            self.revenue_audit_crew = Crew(
                agents=[self.data_ingestion_agent, self.analysis_agent, self.reporting_agent],
                tasks=[self.ingestion_task, self.analysis_task, self.reporting_task],
                verbose=2
            )
            
        except ImportError as e:
            print(f"Error initializing agent system: {e}")
            print("Please install required packages: pip install crewai langchain-google-genai")
            sys.exit(1)
    
    def run_audit(self) -> str:
        """Run the revenue leakage audit
        
        Returns:
            The audit report
        """
        try:
            result = self.revenue_audit_crew.kickoff()
            return result
        except Exception as e:
            error_message = f"Error running audit: {str(e)}"
            print(error_message)
            return error_message


if __name__ == "__main__":
    # Run a test audit
    agent_system = AgentSystem()
    report = agent_system.run_audit()
    print("\nAudit Report:")
    print(report)