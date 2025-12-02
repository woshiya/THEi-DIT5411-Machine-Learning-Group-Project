# Hong Kong Daily Minimum Grass Temperature Forecasting using RNN and LSTM Models

## Project Overview

This project implements and compares Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) models for forecasting Hong Kong's daily minimum grass temperature. The models were trained on historical data from 1980-2024 and evaluated on completely unseen data from January to October 2025.

**Authors:**  
- Waiva Oshiya (220267123)  
- Wong Po Yi (220320080)  
- Jahangir Ashar (220324576)

## Table of Contents
1. [Introduction](#introduction)
2. [Reproducibility](#reproducibility)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Development](#model-development)
6. [Results](#results)
7. [Comparative Analysis](#comparative-analysis)
8. [Error Analysis](#error-analysis)
9. [Installation & Usage](#installation--usage)
10. [Project Structure](#project-structure)

## Introduction

### Project Goal
The aim of this project is to forecast Hong Kong's daily minimum grass temperature using deep learning sequential models. The project compares the performance of simple RNNs versus advanced LSTM architectures in capturing complex temporal patterns in meteorological data.

### Reproducibility
This project consists of 3 main colab notebooks:
- DIT5411_Machine_Learning_Data_PreprocessingGroup_Project.ipynb
- DIT5411_Machine_Learning_Group_Project_RNN.ipynb
- DIT5411_Machine_Learning_LTSM_Group_Project.ipynb

1. Run the Data_Preprocessing notebook first.
This generates the cleaned train.csv and test.csv datasets required for model training.
2. If using Google Colab, upload the generated train.csv and test.csv files into your Colab workspace (the notebook’s working directory) before running the RNN or LSTM notebooks.
3. Next, run the RNN notebook or the LSTM notebook to train and evaluate the respective models.

### Importance of Grass Minimum Temperature
Grass minimum temperature is a critical meteorological indicator that:
- Is often lower than air temperature due to radiative cooling
- Serves as an early warning for frost, road icing, and surface slipperiness
- Supports agricultural planning and urban heat island studies
- Informs public safety measures during cold surges

### Need for Sequential Models
Grass temperature exhibits characteristics that make it suitable for sequential deep learning models:
- Strong daily autocorrelation
- Distinct seasonal cycles (summer peaks: 27–30°C, winter lows: ~5°C)
- Long-term dependencies spanning weeks to months
- Non-linear patterns that challenge traditional statistical methods

## Dataset

### Source
Data is obtained from the Hong Kong Observatory (HKO) through the Hong Kong government's open data portal:
- **URL:** https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-grass-min-temp
- **Location:** Hong Kong Observatory Headquarters (22.3027°N, 114.1742°E)
- **Measurement:** Standard minimum thermometer over short turf, following WMO guidelines

### Data Description
- **Complete Record:** 1 November 1884 to present (one of the longest continuous grass-temperature series globally)
- **Training Period:** 1 January 1980 – 31 December 2024
- **Testing Period:** 1 January 2025 – 30 October 2025 (genuine future data)
- **Format:** CSV with Date index and grass_min_temp (°C) column

### Preprocessing Pipeline
1. Load CSV while skipping header rows
2. Remove non-numeric entries (marked as "***", "#", or "unavailable")
3. Convert valid observations to floating-point numbers
4. Create proper datetime index
5. Restrict to 1980 onwards
6. Resample to strict daily frequency
7. Fill remaining missing values using linear interpolation
8. Split into train.csv (1980-2024) and test.csv (2025)

## Data Preprocessing

### Steps
1. **Loading:** Pandas with `parse_dates=True` and `index_col=0`
2. **Concatenation:** Temporary concatenation of train and test sets for consistent scaling
3. **Sorting:** Chronological ordering by datetime index
4. **Scaling:** MinMaxScaler to range [0, 1], fitted on entire series
5. **Sequence Creation:** Custom `create_sequences()` function for supervised learning

### Sequence Creation Function
```python
def create_sequences(arr, seq_length):
    X, y = [], []
    for i in range(len(arr) - seq_length):
        X.append(arr[i:i + seq_length])
        y.append(arr[i + seq_length])
    return np.array(X), np.array(y)
```

## Hyperparameter Search

### Methodology
We conducted a systematic hyperparameter search to optimize model performance by evaluating three different sequence lengths for both RNN and LSTM architectures:

- **30-day windows**: Captures approximately one month of daily patterns
- **45-day windows**: Represents Hong Kong's typical 4-6 week climate cycles
- **60-day windows**: Extends to two months of historical context

### Sequence Creation Function
```python
def create_sequences(arr, seq_length):
    """
    Converts a time series into supervised learning format
    Args:
        arr: Scaled temperature array
        seq_length: Number of days in input sequence
    Returns:
        X: Input sequences of shape (samples, seq_length, 1)
        y: Target values (next day's temperature)
    """
    X, y = [], []
    for i in range(len(arr) - seq_length):
        X.append(arr[i:i + seq_length])
        y.append(arr[i + seq_length])
    return np.array(X), np.array(y)
```

## Installation & Setup

### Google Colab Environment
This project was developed and tested on Google Colab, which provides free GPU acceleration essential for training deep learning models efficiently.

### Required Libraries

#### Core Deep Learning & Computation
- **TensorFlow 2.x**: Primary deep learning framework for building RNN and LSTM models
- **Keras**: High-level neural networks API (integrated with TensorFlow)
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and time series handling
- **Scikit-learn**: Data preprocessing (MinMaxScaler) and evaluation metrics

#### Visualization & Utilities
- **Matplotlib**: Plotting training curves, predictions, and error distributions
- **Seaborn**: Enhanced statistical visualizations (optional)
- **Datetime**: Date and time manipulation for time series data
