# Restaurant Revenue Predictor

Link to competition: https://www.kaggle.com/c/restaurant-revenue-prediction/overview

The goal of this project is to predict annual restaurant sales based on objective predictors. The root mean squared error (RMSE) is the measure of accuracy. The Kaggle competition provided us with training features, training labels, and testing features. Our training set consists of 137 examples, whereas our testing set consists of 100,000. In each set, there is a combination of known and unknown predictive features.

Data Description:
* Id: The restaurant id.
* Open Date: The opening date of the restaurant.
* City: The city that the restaurant is located in.
* City Group: The city type. Can be one of Big Cities or Other.
* Type: The restaurant type. Can be one of FC: Food Court, IL: Inline, DT: Drive Thru, or MB: Mobile.
* P1 - P37: 37 predictors containing three categories of obfuscated data (demographic, real estate, commercial).

We attempted five different models and techniques:
1. Baseline model (baseline.py)
2. Linear regression using City Group (linear_regression.py)
3. Linear regression with L2 regularization (linear_regression_regularized_.py)
4. Random forest regression (rf_regression.py)
5. 3-layer neural network with k-fold cross validation (neural_network.py)