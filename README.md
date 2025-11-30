# Product Gender Classification Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A comprehensive machine learning project that classifies product gender categories using multiple algorithms including KNN, Decision Trees, Bagged Trees, and Random Forest.

## 📊 Project Overview

This project analyzes e-commerce product data to predict gender categories based on product attributes like price, brand, color, and number of images. The analysis compares five different machine learning models to identify the most accurate approach for product classification.

## 🎯 Business Problem

Accurate product gender classification is crucial for:
- Improved product recommendations
- Targeted marketing campaigns
- Enhanced inventory management
- Better customer experience

## 📁 Dataset

The dataset contains 12,491 products with the following features:
- `ProductID`: Unique product identifier
- `ProductName`: Product description
- `ProductBrand`: Brand name
- `Gender`: Target variable (classification categories)
- `Price (INR)`: Product price in Indian Rupees
- `NumImages`: Number of product images
- `Description`: Product description
- `PrimaryColor`: Dominant product color

## 🧠 Models Implemented

1. **K-Nearest Neighbors (KNN)**
   - With original data
   - With normalized data

2. **Decision Tree**
   - Single decision tree with hyperparameter tuning

3. **Bagged Decision Trees**
   - Ensemble method with bootstrap aggregation

4. **Random Forest**
   - Ensemble method with feature randomness

## 📈 Results Summary

| Model | Test Accuracy | Best Parameters |
|-------|---------------|-----------------|
| KNN (Original) | 70.68% | k=1 |
| KNN (Normalized) | **87.62%** | k=1 |
| Decision Tree | 85.30% | max_depth=None, min_samples_split=5 |
| Bagged Trees | 63.82% | n_estimators=100, max_samples=0.7 |
| Random Forest | 85.86% | n_estimators=100, max_features='sqrt' |

## 🏆 Key Findings

- **Data normalization dramatically improves KNN performance** (16.94% accuracy improvement)
- **KNN with normalized data achieved the highest accuracy** (87.62%)
- **Price and number of images are the most important features**
- **Specific brands show strong gender associations**

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab

### Required Packages

- pandas>=1.5.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scikit-learn>=1.2.0
- jupyter>=1.0.0

## 🔍 Key Visualizations

The project includes comprehensive visualizations:
- Price and image distribution analysis
- Gender category distribution
- Brand and color analysis
- Feature importance charts
- Model performance comparisons
- Decision tree visualizations

## 📝 Methodology

- Data Exploration: Comprehensive EDA with statistical analysis
- Data Preprocessing: Handling missing values, encoding, normalization
- Feature Engineering: One-hot encoding for categorical variables
- Model Training: Cross-validation and hyperparameter tuning
- Model Evaluation: Accuracy, overfitting analysis, feature importance
- Deployment Recommendation: Best model selection with business insights

