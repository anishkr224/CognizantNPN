# Configuration settings for AI Revenue Leakage Detection System

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Vector Database Settings
VECTOR_DB_PATH = os.path.join('models', 'chroma_db')

# Data Paths
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# File Paths
CONTRACTS_FILE = os.path.join(PROCESSED_DATA_DIR, 'contracts.json')
BILLING_FILE = os.path.join(PROCESSED_DATA_DIR, 'billing_records.csv')
USAGE_FILE = os.path.join(PROCESSED_DATA_DIR, 'usage_logs.json')
PROVISIONING_FILE = os.path.join(PROCESSED_DATA_DIR, 'service_provisioning.csv')

# Model Settings
EMBEDDING_MODEL = 'models/embedding-001'
LLM_MODEL = 'gemini-2.5-pro'
LLM_TEMPERATURE = 0.1

# UI Settings
APP_TITLE = 'AI Revenue Leakage Detection System'
APP_DESCRIPTION = 'Proactively identify and prioritize revenue leakage within complex billing workflows'