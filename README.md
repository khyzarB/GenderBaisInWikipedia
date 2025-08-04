# 📊 Wikipedia Gender Bias Analysis
A data science project analyzing and predicting gender bias in Wikipedia biographies to support SDG 5 (Gender Equality).

## 🎯 Project Overview
This project investigates gender representation disparities in Wikipedia by analyzing biographical data across different professions. Using machine learning techniques, we built predictive models to identify bias patterns and created an interactive dashboard to help Wikipedia editors prioritize which biographies need attention.

## 🔍 Problem Statement
Wikipedia shapes global knowledge access, but gender bias in its content can perpetuate societal inequalities. This project:
- Quantifies the extent of gender bias across professions
- Predicts which biographies are at risk of bias
- Provides actionable tools for bias reduction

## 📊 Methodology
### Data Collection

- Gathered biographical data from Wikidata across 5 profession categories
- Collected information on gender, occupation, birth year, and Wikipedia presence
- Final dataset: 1,111 unique biographies after deduplication

### Analysis Pipeline

- Data Processing: Cleaned and standardized profession categories
- Exploratory Analysis: Identified gender distribution patterns
- Feature Engineering: Created STEM classification and temporal groupings
- Machine Learning: Built two Random Forest models
- Visualization: Created an interactive dashboard for insights

### Models Developed
#### Model 1: Article Quality Classifier

Predicts if a biography will have high or low Wikipedia coverage
Features: gender, profession, birth year, STEM classification

#### Model 2: Bias Risk Predictor

Identifies biographies at high risk of gender-based bias
Combines multiple bias indicators into a risk score

## 🖥️ Dashboard Features
The Streamlit dashboard includes three pages:

### 📊 Overview

Key metrics and gender distribution statistics
Comprehensive visualizations of bias patterns
Temporal trends analysis


### 🔍 Bias Predictor

An interactive tool to check bias risk for any profile
Real-time predictions with confidence scores
Combined interpretation of both models


### 📈 Insights & Actions

Highest risk groups identified
Downloadable list of biographies needing attention
Actionable recommendations for Wikipedia editors

## 📈 Key Findings

Significant gender imbalance across all professions analyzed
Quality disparities in Wikipedia article coverage
Certain professions show more pronounced bias patterns
Machine learning models can successfully predict bias risk
Birth year emerged as a surprisingly important factor

## 🛠️ Technologies Used

Languages: Python 3.8+
Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
Machine Learning: Scikit-learn, Random Forest
Web Framework: Streamlit
Data Source: Wikidata Human Gender Indicators
