# Main Application Entry Point for AI Revenue Leakage Detection System

import os
import sys
import streamlit as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import UI
from ui.streamlit_app import main

# Run the Streamlit app
if __name__ == "__main__":
    main()