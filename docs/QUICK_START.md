# Quick Start Guide

## 🚀 Getting Started with HIV Medical Analytics Platform

This guide will help you get the HIV Medical Analytics Platform up and running in minutes.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+**
- **pip** (Python package manager)
- **Git**
- **Virtual environment** (optional but recommended)

---

## Installation

### 1. Clone or Navigate to the Project

```bash
cd "/Users/joelomoroje/HIV Medical Analysis"
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- Data processing: pandas, numpy, polars
- Machine learning: scikit-learn, xgboost, lightgbm
- Visualization: plotly, streamlit, seaborn
- And many more...

---

## Quick Start - 5 Minute Tutorial

### Step 1: Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --n-patients 10000
```

This creates realistic synthetic HIV patient data including:
- **10,000 patient records** with demographics and outcomes
- **Lab results** (CD4 counts, viral loads over time)
- **Treatment regimens** and adherence data

**Output:**
```
data/raw/synthetic_patients.parquet
data/raw/synthetic_lab_results.parquet
data/raw/synthetic_treatments.parquet
```

### Step 2: Run Analytics

```bash
python scripts/run_analytics.py --analysis-type all
```

This performs comprehensive analytics including:
- **Transmission route analysis** by demographics
- **High-risk population identification**
- **Treatment efficacy comparison**
- **Viral suppression factor analysis**

**Output:**
- CSV reports in `reports/generated/`
- Statistical insights in the console

### Step 3: Train Machine Learning Models

```bash
python scripts/train_ml_models.py --model-type xgboost
```

This trains a viral suppression prediction model using:
- **XGBoost** algorithm (or choose: logistic, random_forest, gradient_boosting)
- **80/20 train-test split**
- **Performance metrics** (accuracy, precision, recall, ROC-AUC)

**Output:**
- Trained model saved to `data/models/viral_suppression_model.joblib`
- Feature importance analysis

### Step 4: Launch Interactive Dashboard

```bash
streamlit run src/visualization/dashboards/main_dashboard.py
```

This opens an interactive web dashboard with:
- **Real-time analytics** and visualizations
- **Geographic insights** and heatmaps
- **ML-powered predictions** for new patients
- **Treatment efficacy** comparisons

**Access:** Opens automatically in your browser at `http://localhost:8501`

---

## Key Features Demonstrated

### 1. **Data Engineering Pipeline**

```python
from src.ingestion.data_generator import HIVDataGenerator

# Generate data
generator = HIVDataGenerator(seed=42)
datasets = generator.generate_complete_dataset(n_patients=5000)

# Access datasets
patients = datasets["patients"]
lab_results = datasets["lab_results"]
treatments = datasets["treatments"]
```

### 2. **Transmission Analysis**

```python
from src.analytics.transmission.transmission_analyzer import TransmissionAnalyzer

# Initialize analyzer
analyzer = TransmissionAnalyzer(data=patients)

# Analyze transmission patterns
demo_analysis = analyzer.analyze_transmission_by_demographic()
high_risk = analyzer.identify_high_risk_populations(top_n=10)
risk_scores = analyzer.calculate_transmission_risk_scores()

# Generate report
report = analyzer.generate_summary_report()
```

### 3. **Treatment Efficacy Analysis**

```python
from src.analytics.treatment.treatment_analyzer import TreatmentAnalyzer

# Initialize analyzer
analyzer = TreatmentAnalyzer(
    patients_data=patients,
    lab_results_data=lab_results,
    treatments_data=treatments
)

# Compare treatment regimens
regimens = analyzer.compare_treatment_regimens()

# Analyze viral suppression factors
factors = analyzer.analyze_viral_suppression_factors()

# Calculate effectiveness scores
effectiveness = analyzer.generate_treatment_effectiveness_score()
```

### 4. **Machine Learning Predictions**

```python
from src.ml.models.viral_suppression_predictor import ViralSuppressionPredictor

# Train model
predictor = ViralSuppressionPredictor(model_type="xgboost")
metrics = predictor.train(patients)

# Make predictions
predictions = predictor.predict(new_patients)
probabilities = predictor.predict_proba(new_patients)

# Save/load model
predictor.save_model("model.joblib")
predictor.load_model("model.joblib")
```

---

## Directory Structure

```
HIV-Medical-Analysis/
├── config/                  # Configuration files
│   ├── data_sources.yaml   # Data source configs
│   ├── pipeline_config.yaml # ETL pipeline settings
│   └── ml_config.yaml      # ML model parameters
├── data/                   # Data storage
│   ├── raw/               # Raw data
│   ├── processed/         # Cleaned data
│   ├── analytics/         # Analytics-ready data
│   └── models/            # Trained ML models
├── src/                   # Source code
│   ├── ingestion/         # Data collection
│   ├── etl/              # Data pipeline
│   ├── analytics/         # Advanced analytics
│   ├── ml/               # Machine learning
│   ├── visualization/     # Dashboards
│   └── utils/            # Utilities
├── scripts/              # Automation scripts
├── notebooks/            # Jupyter notebooks
├── tests/               # Unit tests
└── reports/             # Generated reports
```

---

## Next Steps

### Explore Advanced Features

1. **Real-time Data Streaming**
   ```bash
   # Requires Kafka setup
   python src/ingestion/streaming/kafka_consumer.py
   ```

2. **Custom Analytics**
   - Modify `src/analytics/` modules for your specific questions
   - Create new analyzers for additional medical insights

3. **Model Optimization**
   - Tune hyperparameters in `config/ml_config.yaml`
   - Try different algorithms
   - Add new features

4. **API Development**
   - Build REST API for predictions
   - Deploy models to production
   - Create batch prediction pipelines

### Jupyter Notebooks

Explore interactive notebooks:

```bash
jupyter lab
```

Create notebooks in the `notebooks/` directory for:
- Exploratory data analysis
- Custom visualizations
- Ad-hoc queries
- Report generation

---

## Medical Questions Answered

The platform provides insights on critical medical questions:

### 1. **Transmission Analysis**
- What are the most common HIV transmission routes by demographic group?
- Which populations are at highest risk?
- How have transmission patterns changed over time?
- What are geographic variations in transmission routes?

### 2. **Treatment Efficacy**
- What factors predict successful viral suppression?
- How do different ART regimens compare in effectiveness?
- What are the barriers to treatment adherence?
- How quickly do patients achieve viral suppression?

### 3. **Risk Prediction**
- Which patients are at highest risk of treatment failure?
- Can we predict viral suppression likelihood?
- What factors contribute most to positive outcomes?

### 4. **Geographic Patterns**
- Where are new infections rising?
- What are regional differences in treatment outcomes?
- How do social determinants affect HIV care?

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Data Not Found**
```bash
# Generate synthetic data first
python scripts/generate_synthetic_data.py
```

**3. Dashboard Won't Start**
```bash
# Check if streamlit is installed
pip install streamlit

# Run with specific port
streamlit run src/visualization/dashboards/main_dashboard.py --server.port 8502
```

**4. Slow Performance**
```bash
# Reduce dataset size
python scripts/generate_synthetic_data.py --n-patients 1000
```

---

## Support & Resources

- **Documentation**: See `docs/` directory
- **Configuration**: Edit YAML files in `config/`
- **Logs**: Check `logs/` directory for detailed logs
- **Issues**: Review error logs in `logs/errors_*.log`

---

## Sample Output

After running the quick start, you should see output similar to:

```
=== TRANSMISSION ANALYSIS ===
Total Patients: 10,000
Most Common Route: Heterosexual (45.2%)
High-Risk Populations Identified: 10

=== TREATMENT EFFICACY ===
Average Viral Suppression Rate: 84.7%
Most Effective Regimen: TDF/3TC/DTG (89.3% suppression)
Average Time to Suppression: 142 days

=== ML MODEL PERFORMANCE ===
Accuracy:  0.8823
Precision: 0.8956
Recall:    0.8734
ROC-AUC:   0.9412
```

---

## What's Next?

🎉 **Congratulations!** You've successfully set up the HIV Medical Analytics Platform.

Now you can:
- ✅ Explore the interactive dashboard
- ✅ Run custom analytics queries
- ✅ Train your own models
- ✅ Generate comprehensive reports
- ✅ Answer critical medical questions

For advanced usage, see:
- `docs/USER_GUIDE.md` - Detailed usage instructions
- `docs/API.md` - API documentation
- `docs/CONTRIBUTING.md` - Contribution guidelines

---

**Built for Medical Professionals and Data Engineers**

This platform empowers you to:
- Make data-driven medical decisions
- Identify high-risk populations
- Optimize treatment strategies
- Predict patient outcomes
- Monitor program effectiveness

Happy analyzing! 🏥📊🤖

