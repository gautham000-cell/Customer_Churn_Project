# 🚗 Used Vehicle Price Intelligence

A Machine Learning-based web application that predicts the estimated price of a used vehicle based on important vehicle attributes. The project combines data preprocessing, machine learning, and a Streamlit interface to provide an easy-to-use vehicle price prediction system.

## 📌 Project Overview

Buying or selling a used vehicle can be difficult because vehicle prices depend on several factors such as vehicle age, mileage, fuel type, transmission, engine specifications, and other market-related features.

This project uses Machine Learning to analyze historical vehicle data and estimate the expected market price of a used vehicle.

The trained model is integrated into a Streamlit application where users can enter vehicle details and receive a predicted price.

## 🎯 Objectives

* Predict the estimated price of a used vehicle.
* Analyze factors affecting vehicle prices.
* Apply data preprocessing techniques to real-world vehicle data.
* Build and save a Machine Learning prediction model.
* Develop an interactive Streamlit web application.
* Provide a simple interface for users to estimate vehicle prices.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Pickle
* Machine Learning
* Data Preprocessing
* Data Analysis

## 📂 Project Files

```text
Customer_Churn_Project/
│
├── app.py
├── final_vehicle_price_model.pkl
├── vehicle_preprocessor.pkl
├── market_intelligence_data.csv
├── requirements.txt
└── README.md
```

> Note: The repository folder can be renamed to `Used_Vehicle_Price_Intelligence` to better match the actual project.

## 📊 Dataset

The project uses historical vehicle/market data containing information related to used vehicles.

The dataset is processed before being provided to the Machine Learning model.

The preprocessing pipeline handles the transformation of input features into the format required by the trained model.

## 🤖 Machine Learning Model

The project uses a trained Machine Learning regression model to estimate vehicle prices.

The trained model is saved as:

```text
final_vehicle_price_model.pkl
```

The preprocessing pipeline is saved separately as:

```text
vehicle_preprocessor.pkl
```

This allows the Streamlit application to load the trained model and preprocessing pipeline without retraining the model every time the application starts.

## 🔄 Project Workflow

```text
Vehicle Dataset
       ↓
Data Cleaning
       ↓
Data Preprocessing
       ↓
Feature Transformation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Save Model & Preprocessor
       ↓
Streamlit Application
       ↓
Vehicle Price Prediction
```

## 🖥️ Streamlit Application

The application is developed using Streamlit.

Users can enter vehicle-related information through the web interface, and the application processes the input and generates an estimated vehicle price.

The main application file is:

```text
app.py
```

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Open the project folder

```bash
cd YOUR-REPOSITORY
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 📈 Key Features

* Used vehicle price prediction
* Machine Learning-based estimation
* Automated preprocessing
* Interactive Streamlit interface
* Fast prediction using a saved trained model
* Market intelligence data analysis

## 🔮 Future Improvements

* Add more vehicle datasets for better prediction.
* Improve model accuracy through hyperparameter tuning.
* Add multiple Machine Learning models for comparison.
* Add visual analytics and market trends.
* Add vehicle price comparison features.
* Deploy the application online.
* Add downloadable prediction reports.

## 👨‍💻 Author

**Gautham Janus P**

Computer Science Engineering

Interested in Data Science, Machine Learning, Python, and Web Development.

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
