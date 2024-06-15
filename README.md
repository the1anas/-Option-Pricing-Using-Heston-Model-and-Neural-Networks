# Option Pricing Using Heston Model and Neural Networks

This repository contains the implementation of an option pricing model that combines the Heston stochastic volatility model with neural network optimization techniques. The project aims to provide accurate and efficient option pricing by leveraging the strengths of both traditional financial models and modern machine learning algorithms.

## Features
- Data collection using Alpha Vantage and yFinance APIs
- Data preprocessing including log return and volatility calculations
- Exploratory Data Analysis (EDA) with visualization
- Neural network training with Optuna for hyperparameter optimization
- Model validation using the Heston model
- Visualization of predicted vs actual option prices-r requirements.txt
    ```

## Usage
1. **Data Collection:**
    - Collect historical stock data and option market prices using the `data_collection.py` script.
2. **Data Preprocessing:**
    - Preprocess the collected data using the `data_preprocessing.py` script.
3. **Exploratory Data Analysis:**
    - Perform EDA using the `eda.py` script to understand the data distribution and correlations.
4. **Model Training:**
    - Train the neural network model using the `model_training.py` script. Optuna is used for hyperparameter optimization.
5. **Model Validation:**
    - Validate the trained model using the Heston model with the `model_validation.py` script.

## Results
- Visualizations of the model performance and comparisons between predicted and actual option prices.
- Example graphs showing the predicted vs actual option prices.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.



