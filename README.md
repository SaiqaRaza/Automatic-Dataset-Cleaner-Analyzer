# Automatic-Dataset-Cleaner & Analyzer

An interactive web-based tool that helps you clean, preprocess, and prepare datasets for analysis or machine learning.  
Built with **Python** and **Streamlit**, it analyzes your dataset, detects potential data quality issues, recommends preprocessing techniques, and allows you to apply them with just a few clicks.

---

## 🚀 Features
- Upload CSV datasets and preview data instantly.
- Detect and handle missing values with custom strategies.
- Identify and process high-cardinality categorical columns:
  - Drop
  - Frequency encoding
  - Hash encoding
- Encode low-cardinality categorical columns:
  - One-Hot Encoding
  - Label Encoding
- Scale numeric features:
  - StandardScaler
  - MinMaxScaler
  - None
- Generate data visualizations:
  - Histograms
  - Boxplots
  - Correlation heatmaps
- Produce:
  - Cleaned dataset (`cleaned_dataset.csv`)
  - Preprocessing report (`preprocessing_report.txt`)
  - Visual plots
- Download results as a single ZIP file.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/automatic-dataset-cleaner.git
cd automatic-dataset-cleaner

# Install dependencies
pip install -r requirements.txt

