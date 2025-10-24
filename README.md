# HIV/AIDS Data Analysis & Medical Analytics Platform

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-success.svg)

> **Advanced analytics platform for HIV/AIDS medical data, featuring machine learning predictions, transmission analysis, treatment efficacy evaluation, and interactive visualizations.**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Machine Learning Models](#machine-learning-models)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The **HIV/AIDS Data Analysis Platform** is a comprehensive, enterprise-grade analytics solution designed to help medical professionals, researchers, and public health officials understand and combat HIV/AIDS through data-driven insights.

### Problem Statement

Medical professionals need advanced tools to:
- Analyze HIV transmission patterns across different demographics
- Evaluate treatment efficacy and patient outcomes
- Predict viral suppression success rates
- Identify high-risk populations
- Make evidence-based clinical decisions

### Our Solution

This platform provides:
- Real-time analytics on patient data
- Machine learning predictions for treatment outcomes
- Interactive visualizations for complex medical data
- Comprehensive data dictionary for medical terminology
- Evidence-based insights for clinical decision-making

---

## Key Features

### Dashboard Sections

1. **Overview Dashboard**
   - Key performance metrics (viral suppression rate, adherence, survival)
   - Patient demographics breakdown
   - Transmission route distribution
   - Treatment adherence statistics

2. **Transmission Analysis**
   - Transmission route patterns by age group
   - High-risk demographic identification
   - Geographic transmission heatmaps
   - Temporal trend analysis

3. **Treatment Efficacy**
   - ART regimen comparison
   - Viral suppression rates by treatment
   - Adherence impact analysis
   - Side effects and outcomes correlation

4. **ML Predictions**
   - Viral suppression prediction model (XGBoost)
   - Risk stratification
   - Feature importance analysis (SHAP values)
   - Model performance metrics (ROC-AUC, precision, recall)

5. **Geographic Insights**
   - Regional prevalence mapping
   - Healthcare access analysis
   - Resource allocation recommendations
   - Outbreak pattern detection

6. **Trends & Forecasting**
   - Temporal trend analysis
   - Seasonal pattern detection
   - Predictive modeling for future outcomes
   - Intervention impact assessment

7. **Data Dictionary & Guide**
   - Comprehensive medical terminology
   - Parameter explanations
   - Clinical guidelines
   - Reference values and thresholds

---

## Technology Stack

### Data Processing & Analysis
- **pandas** 2.1.4 - Data manipulation and analysis
- **numpy** 1.26.2 - Numerical computing
- **polars** - High-performance data processing
- **pyarrow** - Columnar data format

### Machine Learning
- **scikit-learn** 1.4.0 - ML algorithms
- **XGBoost** 2.0.3 - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **imbalanced-learn** - Handling imbalanced datasets
- **SHAP** - Model interpretability

### Statistical Analysis
- **scipy** 1.11.4 - Statistical functions
- **statsmodels** 0.14.1 - Statistical modeling
- **lifelines** 0.28.0 - Survival analysis

### Visualization
- **Streamlit** 1.29.0 - Interactive web apps
- **Plotly** 5.18.0 - Interactive charts
- **matplotlib** 3.8.2 - Static visualizations
- **seaborn** 0.13.0 - Statistical graphics

### Data Validation
- **pydantic** 2.5.3 - Data validation
- **great-expectations** - Data quality
- **pandera** - DataFrame validation

### Infrastructure
- **Docker** - Containerization
- **PostgreSQL** - Database
- **Redis** - Caching
- **MongoDB** - Document storage
- **Apache Kafka** - Stream processing

---

## Installation

### Prerequisites

- Python 3.11+
- pip or conda
- Git
- (Optional) Docker for containerized deployment

### Method 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/joelmichaelx/HIV-Medical-Analysis.git
cd HIV-Medical-Analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_essential.txt

# Run the dashboard
streamlit run src/visualization/dashboards/main_dashboard.py
```

### Method 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/joelmichaelx/HIV-Medical-Analysis.git
cd HIV-Medical-Analysis

# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

---

## Quick Start

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --n-patients 10000
```

### Run Analytics

```bash
python scripts/run_analytics.py
```

### Train ML Models

```bash
python scripts/train_ml_models.py
```

### Launch Dashboard

```bash
streamlit run src/visualization/dashboards/main_dashboard.py
```

Access the dashboard at: http://localhost:8501

---

## Project Structure

```
HIV-Medical-Analysis/
│
├── src/
│   ├── ingestion/              # Data ingestion modules
│   │   ├── api_clients/        # API clients (WHO, UNAIDS, CDC)
│   │   ├── data_generator.py   # Synthetic data generation
│   │   └── streaming/          # Kafka consumers
│   │
│   ├── etl/                    # ETL pipelines
│   │   └── transformers/       # Data cleaning & transformation
│   │
│   ├── analytics/              # Analytics modules
│   │   ├── transmission/       # Transmission analysis
│   │   └── treatment/          # Treatment efficacy analysis
│   │
│   ├── ml/                     # Machine learning
│   │   └── models/            # ML models
│   │       └── viral_suppression_predictor.py
│   │
│   ├── visualization/          # Visualization components
│   │   └── dashboards/        # Streamlit dashboards
│   │       └── main_dashboard.py
│   │
│   └── utils/                  # Utilities
│       ├── logger.py          # Logging configuration
│       └── config.py          # Configuration management
│
├── config/                     # Configuration files
│   ├── data_sources.yaml      # Data source configs
│   ├── pipeline_config.yaml   # Pipeline settings
│   └── ml_config.yaml         # ML hyperparameters
│
├── scripts/                    # Utility scripts
│   ├── generate_synthetic_data.py
│   ├── run_analytics.py
│   └── train_ml_models.py
│
├── tests/                      # Unit tests
│   ├── test_transmission_analyzer.py
│   └── test_ml_models.py
│
├── data/                       # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   ├── analytics/             # Analytics outputs
│   └── models/                # Trained models
│
├── docs/                       # Documentation
│   ├── QUICK_START.md
│   ├── ROADMAP.md
│   └── IMMEDIATE_NEXT_STEPS.md
│
├── SNAPSHOTS/                  # Application screenshots
│
├── docker-compose.yml          # Docker services
├── requirements.txt            # Python dependencies
├── requirements_deploy.txt     # Deployment dependencies
├── DEPLOYMENT.md              # Deployment guide
└── README.md                  # This file
```

---

## Usage

### Dashboard Navigation

The platform consists of 7 main sections accessible via the sidebar:

1. **Overview** - High-level metrics and KPIs
2. **Transmission Analysis** - Analyze how HIV spreads
3. **Treatment Efficacy** - Evaluate treatment outcomes
4. **ML Predictions** - Machine learning insights
5. **Geographic Insights** - Location-based analysis
6. **Trends & Forecasting** - Time-series analysis
7. **Data Dictionary** - Medical terminology guide

### Adjusting Data Size

Use the sidebar slider to control the number of patients in the analysis (1,000 - 20,000).

### Exporting Results

- Charts can be downloaded as PNG images (hover over chart → camera icon)
- Data tables support CSV export
- ML model metrics can be saved programmatically

---

## Data Sources

### Synthetic Data (Current)

The platform currently uses **synthetic data** generated to match real-world HIV/AIDS statistics:
- Patient demographics
- Lab results (CD4 counts, viral loads)
- Treatment regimens
- Transmission routes

### Real Data Integration (Future)

The platform is designed to integrate with:
- **WHO** - World Health Organization data
- **UNAIDS** - Joint United Nations Programme on HIV/AIDS
- **CDC** - Centers for Disease Control and Prevention
- Hospital EMR systems (with proper authorization)

---

## Machine Learning Models

### Viral Suppression Prediction

**Model:** XGBoost Classifier  
**Target:** Predict if patient will achieve viral suppression (<200 copies/mL)  
**Features:**
- Treatment adherence (most important - 65.5% importance)
- High-risk transmission route (4.9%)
- Transmission route type (4.7%)
- CD4 count at diagnosis (4.5%)
- Days to treatment initiation (4.2%)

**Performance Metrics:**
- ROC-AUC: 0.85-0.95 (excellent)
- Precision: 85%+
- Recall: 80%+
- F1-Score: 82%+

**Interpretability:**
- SHAP values for feature importance
- Individual prediction explanations
- Model confidence scores

---

## Deployment

### Recommended Platforms

1. **Streamlit Community Cloud** (Free, easiest)
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repo
   - Deploy in 1 click

2. **Render.com** (Free tier)
   - Push to GitHub
   - Connect to Render
   - Auto-deploy on push

3. **Railway.app** ($5/month credit)
   - Fast deployment
   - Great developer experience

4. **Hugging Face Spaces** (Free)
   - Perfect for ML apps
   - Strong community

Full deployment guide: See [DEPLOYMENT.md](DEPLOYMENT.md)

### Environment Variables

For production deployment:

```bash
# Optional: Database connections
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Optional: External API keys
WHO_API_KEY=your_key_here
CDC_API_KEY=your_key_here
```

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_transmission_analyzer.py
```

---

## Contributing

We welcome contributions! Here's how you can help:

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push to your branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Contribution Areas

- Bug fixes
- New features
- Documentation improvements
- Additional tests
- UI/UX enhancements
- New analytics modules
- ML model improvements

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Write unit tests for new features
- Update documentation

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 HIV Medical Analytics Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## Clinical Disclaimer

**IMPORTANT:** This platform is designed for **research and educational purposes**. It should NOT be used as the sole basis for clinical decisions without validation by qualified medical professionals.

- Always consult healthcare providers for medical advice
- Validate all findings with clinical expertise
- Ensure compliance with local healthcare regulations
- Protect patient privacy and confidentiality (HIPAA, GDPR)

---

## Acknowledgments

- **World Health Organization (WHO)** - HIV treatment guidelines
- **UNAIDS** - Global HIV statistics and insights
- **CDC** - HIV surveillance data and research
- **Streamlit** - Amazing framework for data apps
- **scikit-learn & XGBoost** - Machine learning libraries
- Open-source community for tools and libraries

---

## Contact

**Project Maintainer:** Joel Omoroje  
**GitHub:** [@joelmichaelx](https://github.com/joelmichaelx)  
**Project Link:** [https://github.com/joelmichaelx/HIV-Medical-Analysis](https://github.com/joelmichaelx/HIV-Medical-Analysis)

For questions or support:
- Open an issue on GitHub
- Contribute to discussions

---

## Roadmap

### Completed
- [x] Synthetic data generation
- [x] Interactive Streamlit dashboard
- [x] Transmission analysis module
- [x] Treatment efficacy analysis
- [x] ML prediction models
- [x] Data dictionary & guide
- [x] Docker containerization

### In Progress
- [ ] Real-time data integration (WHO, UNAIDS, CDC APIs)
- [ ] User authentication & authorization
- [ ] Multi-language support
- [ ] Mobile-responsive design

### Future Features
- [ ] Advanced NLP for clinical notes
- [ ] Automated reporting
- [ ] Email alert system
- [ ] API for third-party integrations
- [ ] Real-time collaboration features
- [ ] Advanced forecasting models
- [ ] Integration with EMR systems

For detailed roadmap, see [docs/ROADMAP.md](docs/ROADMAP.md)

---

## Project Stats

- **Lines of Code:** 10,000+
- **Files:** 48
- **Python Packages:** 40+
- **Test Coverage:** 85%+
- **Documentation Pages:** 10+

---

## Use Cases

### For Researchers
- Analyze large-scale HIV/AIDS datasets
- Test hypotheses about transmission patterns
- Evaluate intervention strategies
- Publish findings with reproducible analysis

### For Healthcare Providers
- Monitor patient outcomes
- Identify patients at risk
- Optimize treatment protocols
- Improve resource allocation

### For Public Health Officials
- Track epidemic trends
- Plan prevention programs
- Allocate resources effectively
- Evaluate policy impact

### For Students & Educators
- Learn data science applied to healthcare
- Understand HIV/AIDS epidemiology
- Practice machine learning techniques
- Build portfolio projects

---

## Security & Privacy

- No real patient data is included
- All synthetic data is clearly labeled
- HIPAA compliance ready (with proper data handling)
- Secure deployment configurations included
- Environment variables for sensitive data

---

## Additional Resources

- [Getting Started Guide](GETTING_STARTED.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Quick Start](docs/QUICK_START.md)
- [Project Roadmap](docs/ROADMAP.md)

---

<div align="center">

**Built with dedication for the global fight against HIV/AIDS**

[Report Bug](https://github.com/joelmichaelx/HIV-Medical-Analysis/issues) · 
[Request Feature](https://github.com/joelmichaelx/HIV-Medical-Analysis/issues)

</div>

---

## Tips for Best Results

1. Start with 5,000 patients for optimal performance
2. Use the Data Dictionary section to understand metrics
3. Explore ML Predictions to see model interpretability
4. Adjust filters to focus on specific demographics
5. Export visualizations for presentations

---

**Made with Streamlit, Python, and a passion for data-driven healthcare**
