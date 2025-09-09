# Knowledge Base and RAG Setup for AI Revenue Leakage Detection System

import os
import json
import pandas as pd
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config
from config import VECTOR_DB_PATH, PROCESSED_DATA_DIR

class KnowledgeBase:
    """Knowledge Base for AI Revenue Leakage Detection System using RAG"""
    
    def __init__(self):
        """Initialize the knowledge base"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from config import EMBEDDING_MODEL, GEMINI_API_KEY
            import google.generativeai as genai
            
            # Set the API key for Google Generative AI
            genai.configure(api_key=GEMINI_API_KEY)
            
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
            
            # Create vector store directory if it doesn't exist
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            
            # Initialize vector store
            self.vector_db = None
        except ImportError as e:
            print(f"Error initializing knowledge base: {e}")
            print("Please install required packages: pip install langchain langchain-google-genai chromadb")
            sys.exit(1)
    
    def load_contracts(self) -> List[Dict[str, Any]]:
        """Load contract data from file"""
        contracts_file = os.path.join(PROCESSED_DATA_DIR, "contracts.json")
        if not os.path.exists(contracts_file):
            print(f"Contracts file not found: {contracts_file}")
            return []
        
        with open(contracts_file, 'r') as f:
            contracts = json.load(f)
        
        return contracts
    
    def load_billing_records(self) -> pd.DataFrame:
        """Load billing records from file"""
        billing_file = os.path.join(PROCESSED_DATA_DIR, "billing_records.csv")
        if not os.path.exists(billing_file):
            print(f"Billing records file not found: {billing_file}")
            return pd.DataFrame()
        
        return pd.read_csv(billing_file)
    
    def load_usage_logs(self) -> List[Dict[str, Any]]:
        """Load usage logs from file"""
        usage_file = os.path.join(PROCESSED_DATA_DIR, "usage_logs.json")
        if not os.path.exists(usage_file):
            print(f"Usage logs file not found: {usage_file}")
            return []
        
        with open(usage_file, 'r') as f:
            usage_logs = json.load(f)
        
        return usage_logs
    
    def load_service_provisioning(self) -> pd.DataFrame:
        """Load service provisioning records from file"""
        provisioning_file = os.path.join(PROCESSED_DATA_DIR, "service_provisioning.csv")
        if not os.path.exists(provisioning_file):
            print(f"Service provisioning file not found: {provisioning_file}")
            return pd.DataFrame()
        
        return pd.read_csv(provisioning_file)
    
    def create_vector_store(self):
        """Create vector store from all data sources"""
        # Load all data
        contracts = self.load_contracts()
        billing_records = self.load_billing_records()
        usage_logs = self.load_usage_logs()
        service_provisioning = self.load_service_provisioning()
        
        # Convert to text for embedding
        contracts_text = json.dumps(contracts, indent=2)
        billing_text = billing_records.to_string()
        usage_text = json.dumps(usage_logs, indent=2)
        provisioning_text = service_provisioning.to_string()
        
        # Combine all text
        all_text = (
            "=== CONTRACTS ===\n" + contracts_text + "\n\n" +
            "=== BILLING RECORDS ===\n" + billing_text + "\n\n" +
            "=== USAGE LOGS ===\n" + usage_text + "\n\n" +
            "=== SERVICE PROVISIONING ===\n" + provisioning_text
        )
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(all_text)
        
        # Create vector store
        from langchain_community.vectorstores import Chroma
        self.vector_db = Chroma.from_texts(
            chunks, 
            self.embeddings, 
            persist_directory=VECTOR_DB_PATH
        )
        self.vector_db.persist()
        
        print(f"Vector store created with {len(chunks)} chunks")
        return self.vector_db
    
    def load_vector_store(self):
        """Load existing vector store"""
        from langchain_community.vectorstores import Chroma
        
        if not os.path.exists(VECTOR_DB_PATH):
            print(f"Vector store not found: {VECTOR_DB_PATH}")
            return None
        
        self.vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=self.embeddings)
        print("Vector store loaded successfully")
        return self.vector_db
    
    def get_vector_store(self):
        """Get vector store, creating it if it doesn't exist"""
        if self.vector_db is None:
            try:
                self.vector_db = self.load_vector_store()
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Creating new vector store...")
                self.vector_db = self.create_vector_store()
        
        return self.vector_db
    
    def similarity_search(self, query, k=5):
        """Search for similar documents in the vector store"""
        vector_db = self.get_vector_store()
        if vector_db is None:
            print("Vector store not available")
            return []
        
        docs = vector_db.similarity_search(query, k=k)
        return docs


if __name__ == "__main__":
    # Create knowledge base
    kb = KnowledgeBase()
    
    # Create vector store
    vector_db = kb.create_vector_store()
    
    # Test similarity search
    query = "What is the agreed rate for cloud storage?"
    docs = kb.similarity_search(query)
    
    print(f"\nSearch results for: {query}")
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:\n{doc.page_content[:200]}...")