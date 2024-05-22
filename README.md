# Sendy ETA Prediction Project

## Overview

This project aims to predict the Estimated Time of Arrival (ETA) for orders placed on the Sendy platform. The dataset includes order details and rider metrics based on orders made using Sendy's API, web, and mobile applications. Accurate ETA predictions can enhance customer satisfaction and optimize delivery operations.

## Project Objective

The primary objective is to develop a predictive model to estimate the ETA from pick-up to drop-off for Sendy orders. 

This involves:
1. Data exploration and preprocessing.
2. Feature engineering.
3. Model development and evaluation.
4. Integration and optimization for real-time predictions.

## Datasets

The following datasets are used:
- `Train.csv`: Training dataset with order details.
- `Test.csv`: Test dataset for model evaluation.
- `Riders.csv`: Rider metrics.
- `VariableDefinitions.csv`: Definitions of the variables in the datasets.

### 1. Data Exploration and Understanding

The datasets were explored to understand their structure and contents. 

The key variables include:
- Order details (Order No, User Id, Vehicle Type, etc.)
- Time stamps (Placement Time, Confirmation Time, Pickup Time, etc.)
- Location details (Pickup Latitude and Longitude, Destination Latitude and Longitude)
- Rider metrics (Number of Orders, Age, Average Rating, etc.)

### 2. Data Cleaning and Preprocessing

The data cleaning and preprocessing steps involved:
- Converting time columns to datetime format.
- Filling missing values for temperature and precipitation using the Open-Meteo API.
- Ensuring the date format and range are correct for API requests.
- Handling cases where weather data might be missing or unavailable.

### 3. Fetching Weather Data

To enhance the model's accuracy, historical weather data (temperature and precipitation) was fetched from the Open-Meteo API. 

The steps include:
- Setting up the Open-Meteo API client with caching and retry mechanisms.
- Converting order placement times to Unix timestamps and ensuring they fall within the allowed date range.
- Fetching the closest available three-hour interval weather data based on the order placement time.
- Filling missing values in the dataset with the fetched weather data.

### 4. Integrating Weather Data

The weather data (temperature and precipitation) was integrated into the dataset to fill missing values, improving the overall data quality for model training.

## Usage

### Prerequisites

Ensure the following Python packages are installed:
```
pip install openmeteo-requests requests-cache retry-requests numpy pandas
```