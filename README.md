# Advanced HIV Medical Analytics Platform

## Overview
A comprehensive data engineering and analytics platform for HIV medical research, built to answer critical questions that medical professionals face. This system processes real-time and historical data from multiple authoritative sources to provide actionable insights.

## Key Features

### 1. Data Engineering Pipeline
- **Multi-Source Data Ingestion**: Automated collection from WHO, UNAIDS, CDC, and clinical databases
- **Real-Time Streaming**: Apache Kafka integration for live data processing
- **ETL Pipeline**: Robust extraction, transformation, and loading with data quality checks
- **Data Lake Architecture**: Organized storage with bronze, silver, and gold layers

### 2. Advanced Analytics
Answers critical medical questions including:
- **Transmission Analysis**: Which transmission routes are most prevalent by region, age, gender?
- **Treatment Efficacy**: What are the survival rates and viral suppression patterns?
- **Geographic Hotspots**: Where are new infections rising? What are the risk factors?
- **Demographic Patterns**: Which populations are most vulnerable?
- **Drug Resistance**: Tracking antiretroviral resistance patterns
- **Prevention Impact**: How effective are PrEP and other prevention strategies?
- **Co-morbidity Analysis**: HIV interactions with TB, Hepatitis, COVID-19

### 3. Machine Learning Models
- **Risk Prediction**: Identify high-risk populations
- **Outbreak Forecasting**: Predict geographic spread patterns
- **Treatment Response**: Predict patient outcomes based on demographics and treatment
- **Clustering**: Patient segmentation for targeted interventions

### 4. Visualization & Reporting
- **Interactive Dashboards**: Real-time metrics for medical professionals
- **Geographic Heatmaps**: Visual representation of infection rates
- **Trend Analysis**: Time-series visualizations
- **Automated Reports**: PDF/HTML generation for stakeholders

## Technology Stack

### Core Infrastructure
- **Python 3.11+**: Primary programming language
- **Apache Kafka**: Real-time data streaming
- **PostgreSQL**: Structured data storage
- **MongoDB**: Semi-structured clinical data
- **Redis**: Caching and session management

### Data Processing
- **Pandas & Polars**: Data manipulation
- **PySpark**: Large-scale data processing
- **Airflow**: Workflow orchestration
- **dbt**: Data transformation

### Machine Learning
- **Scikit-learn**: Classical ML algorithms
- **XGBoost/LightGBM**: Gradient boosting models
- **TensorFlow/PyTorch**: Deep learning for complex predictions
- **SHAP**: Model interpretability

### Visualization
- **Plotly/Dash**: Interactive dashboards
- **Streamlit**: Rapid prototyping and data apps
- **Seaborn/Matplotlib**: Statistical visualizations
- **Folium**: Geographic mapping

### DevOps & Monitoring
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Prometheus/Grafana**: Monitoring
- **Great Expectations**: Data quality
- **Pytest**: Testing framework

## Project Structure

```
HIV-Medical-Analysis/
├── config/                     # Configuration files
│   ├── data_sources.yaml      # Data source configurations
│   ├── pipeline_config.yaml   # ETL pipeline settings
│   └── ml_config.yaml         # ML model parameters
├── data/                      # Data storage (gitignored)
│   ├── raw/                   # Bronze layer - raw data
│   ├── processed/             # Silver layer - cleaned data
│   ├── analytics/             # Gold layer - analytics-ready
│   └── models/                # Trained ML models
├── src/
│   ├── ingestion/             # Data collection modules
│   │   ├── api_clients/       # API integrations
│   │   ├── streaming/         # Kafka consumers/producers
│   │   └── batch/             # Batch data loaders
│   ├── etl/                   # ETL pipeline
│   │   ├── extractors/        # Data extraction
│   │   ├── transformers/      # Data transformation
│   │   ├── loaders/           # Data loading
│   │   └── validators/        # Data quality checks
│   ├── analytics/             # Advanced analytics
│   │   ├── transmission/      # Transmission analysis
│   │   ├── treatment/         # Treatment efficacy
│   │   ├── geographic/        # Geographic analysis
│   │   └── demographic/       # Demographic patterns
│   ├── ml/                    # Machine learning
│   │   ├── models/            # Model definitions
│   │   ├── training/          # Training pipelines
│   │   ├── inference/         # Prediction services
│   │   └── evaluation/        # Model evaluation
│   ├── visualization/         # Visualization components
│   │   ├── dashboards/        # Dashboard applications
│   │   ├── reports/           # Report generators
│   │   └── charts/            # Chart components
│   └── utils/                 # Utility functions
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit and integration tests
├── scripts/                   # Automation scripts
├── docs/                      # Documentation
├── airflow/                   # Airflow DAGs
└── docker/                    # Docker configurations

```

## Quick Start

### Prerequisites
```bash
# Python 3.11+
python --version

# Docker and Docker Compose
docker --version
docker-compose --version
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd HIV-Medical-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/example.env .env
# Edit .env with your API keys and credentials

# Initialize databases
python scripts/init_databases.py

# Run data pipeline
python scripts/run_pipeline.py
```

### Running with Docker
```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## Key Medical Questions Answered

### 1. Transmission Route Analysis
**Question**: "What are the most common HIV transmission routes by demographic group?"
- Analyze transmission patterns across age groups, genders, regions
- Track changes over time
- Identify high-risk behaviors

### 2. Treatment Outcomes
**Question**: "What factors predict successful viral suppression?"
- Analyze treatment adherence patterns
- Compare antiretroviral regimens
- Identify barriers to treatment success

### 3. Geographic Spread
**Question**: "Where are new infections rising, and why?"
- Hotspot detection algorithms
- Correlation with social determinants of health
- Migration pattern analysis

### 4. Prevention Effectiveness
**Question**: "How effective are current prevention strategies?"
- PrEP uptake and effectiveness
- Condom distribution impact
- Education program outcomes

### 5. Co-morbidity Patterns
**Question**: "How do co-infections affect HIV outcomes?"
- TB-HIV co-infection analysis
- Hepatitis C interactions
- COVID-19 impact on HIV care

### 6. Drug Resistance
**Question**: "What are the emerging drug resistance patterns?"
- Genotype-phenotype analysis
- Resistance mutation tracking
- Treatment failure prediction

## Data Sources

### Primary Sources
1. **WHO Global Health Observatory**: https://www.who.int/data/gho
2. **UNAIDS Data**: https://aidsinfo.unaids.org/
3. **CDC HIV Statistics**: https://www.cdc.gov/hiv/statistics/
4. **PEPFAR Dashboards**: https://data.pepfar.gov/
5. **NIH HIV/AIDS Database**: https://clinicalinfo.hiv.gov/

### Synthetic Data
For testing and development, we generate realistic synthetic datasets that maintain statistical properties of real data while ensuring privacy.

## Data Quality & Privacy

### Data Quality Framework
- **Completeness Checks**: Ensure required fields are populated
- **Consistency Validation**: Cross-reference data across sources
- **Accuracy Monitoring**: Statistical anomaly detection
- **Timeliness Tracking**: Monitor data freshness

### Privacy & Security
- **De-identification**: All patient data is anonymized
- **HIPAA Compliance**: Healthcare data protection standards
- **Encryption**: Data encrypted at rest and in transit
- **Access Control**: Role-based access management
- **Audit Logging**: Comprehensive activity tracking

## Usage Examples

### Running Analytics Queries
```python
from src.analytics.transmission import TransmissionAnalyzer

analyzer = TransmissionAnalyzer()

# Analyze transmission routes by region
results = analyzer.analyze_transmission_routes(
    region='Sub-Saharan Africa',
    time_period='2020-2024',
    demographic_breakdown=True
)

print(results.summary())
results.visualize()
```

### Training ML Models
```python
from src.ml.training import RiskPredictionTrainer

trainer = RiskPredictionTrainer()
model = trainer.train(
    data_path='data/processed/risk_factors.parquet',
    model_type='gradient_boosting'
)

# Evaluate model
metrics = trainer.evaluate(model)
print(f"AUC-ROC: {metrics['auc_roc']}")
```

### Creating Dashboards
```bash
# Launch Streamlit dashboard
streamlit run src/visualization/dashboards/main_dashboard.py

# Access at http://localhost:8501
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_analytics/
```

## Contributing

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- World Health Organization (WHO)
- Joint United Nations Programme on HIV/AIDS (UNAIDS)
- Centers for Disease Control and Prevention (CDC)
- Global health research community

## Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**⚠️ Disclaimer**: This platform is for research and educational purposes. Medical decisions should always be made by qualified healthcare professionals.

