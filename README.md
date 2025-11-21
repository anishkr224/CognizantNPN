# AI Revenue Leakage Detection System

An end-to-end Python-based AI system that proactively identifies, prioritizes, and explains revenue leakage points within complex billing workflows by analyzing disparate datasets.

## Project Overview

This system is designed to detect discrepancies in billing workflows such as:
- Missing charges
- Incorrect rates
- Usage mismatches
- Duplicate entries

It provides near real-time alerts, automated ticket creation, root cause analysis insights, and helps reduce manual audit efforts while accelerating revenue recovery.

## Technology Stack

- **Framework**: CrewAI (Agentic AI Orchestration)
- **LLM**: Google Gemini API (or OpenAI GPT-4)
- **Vector Database**: ChromaDB
- **Backend**: Python, LangChain, LlamaIndex
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, PyMuPDF/Tesseract (OCR)
- **Deployment**: Docker, AWS SageMaker/Google Cloud Run

## Project Structure

```
.
├── agents/                # CrewAI agent system components
│   ├── agents.py         # Agent definitions and crew setup
│   └── tools.py          # Custom tools for agents
├── data/                  # Data storage directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── models/                # Vector database and model files
├── ui/                    # Streamlit UI code
│   └── streamlit_app.py  # Streamlit application
├── utils/                 # Utility functions
│   ├── data_generator.py # Synthetic data generation
│   ├── evaluation.py     # Evaluation metrics
│   └── knowledge_base.py # RAG system setup
├── validation_results/    # Validation outputs
├── .env.example          # Environment variables template
├── app.py                # Main application entry point
├── config.py             # Configuration settings
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Project dependencies
├── validate_system.py    # System validation script
└── README.md             # Project documentation
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)

### Environment Setup

1. Clone the repository
2. Create a `.env` file
3. Add your API keys to the `.env` file

### Installation

#### Local Development

```bash
pip install -r requirements.txt
python app.py
```

#### Docker Deployment

```bash
docker-compose up -d
```

This will build and start the containerized application, making it accessible at http://localhost:8501.

## Usage

### Running the Application

```bash
python app.py
```

Or access the Streamlit UI directly:

```bash
streamlit run ui/streamlit_app.py
```

### Validating System Performance

```bash
python validate_system.py
```

### Workflow

1. **Data Ingestion**: Upload billing, contract, usage, and service data or generate synthetic data through the UI.
2. **Knowledge Base Creation**: System processes and indexes the data for efficient retrieval.
3. **Analysis**: CrewAI agents analyze the data to detect revenue leakage issues.
4. **Reporting**: Results are presented with detailed findings and financial impact.

## Evaluation Metrics

The system includes comprehensive evaluation metrics:

- **Precision**: % of correctly flagged discrepancies out of total flags
- **Recall**: % of actual discrepancies found out of total existing ones
- **F1 Score**: Balance between precision and recall
- **Confusion Matrix**: Visualization of true/false positives and negatives
- **Financial Impact Analysis**: Comparison of detected vs. actual leakage amounts
- **Recovery Rate**: Percentage of true leakage amount successfully detected
- **Estimated Revenue Recovery**: Sum of((agreed_rate - billed_rate) * usage_quantity) for all caught errors

## Deployment

The system can be deployed using Docker and Docker Compose for scalability and ease of management. The provided configuration includes:

- Containerized application with all dependencies
- Volume mapping for persistent data storage
- Environment variable configuration
- Port mapping for web interface access
