import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.baseline_model import baseline_model
from models.linear_regression import simple_lin_reg_model
from models.linear_regression_regularized import lin_reg_l2_model
from models.rf_regression import rf_model
from models.neural_network import nn_model

# One hot encoding
def one_hot(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return res

# Pre-process the data
def pre_process(training_df, testing_df):

    # Convert all columns with a space to an underscore
    training_df.columns = training_df.columns.str.replace(' ', '_')
    testing_df.columns = testing_df.columns.str.replace(' ', '_')

    # One hot encode training and testing data
    training_df = one_hot(training_df, 'City_Group')
    training_df = one_hot(training_df, 'Type')
    testing_df = one_hot(testing_df, 'City_Group')
    testing_df = one_hot(testing_df, 'Type')

    # Process the open date to find number of days a restaurant has been open
    training_df['Open_Date'] = pd.to_datetime(training_df['Open_Date'], format='%m/%d/%Y')
    training_df['Curr_Date'] = pd.to_datetime('05/11/2021')
    training_df['Days_Open'] = training_df['Curr_Date'] - training_df['Open_Date']
    training_df['Days_Open'] = training_df['Days_Open'] / pd.Timedelta(1, unit='d')  # convert to int

    testing_df['Open_Date'] = pd.to_datetime(testing_df['Open_Date'], format='%m/%d/%Y')
    testing_df['Curr_Date'] = pd.to_datetime('05/11/2021')
    testing_df['Days_Open'] = testing_df['Curr_Date'] - testing_df['Open_Date']
    testing_df['Days_Open'] = testing_df['Days_Open'] / pd.Timedelta(1, unit='d')  # convert to int

    # Normalize the number of days open
    max_num_days = training_df['Days_Open'].max()
    training_df['Days_Open'] = training_df['Days_Open'] / max_num_days
    testing_df['Days_Open'] = testing_df['Days_Open'] / max_num_days

    # Drop columns
    training_df = training_df.drop(columns=['Id', 'Open_Date', 'Curr_Date', 'City', 'City_Group', 'Type', 'revenue'])
    testing_df = testing_df.drop(columns=['Id', 'Open_Date', 'Curr_Date', 'City', 'City_Group', 'Type', 'Type_MB'])

    # Scale the remaining training and testing data and convert to numpy arrays
    scaler = StandardScaler()
    X_train = scaler.fit_transform(training_df)
    X_test = scaler.fit_transform(testing_df)

    return X_train, y_train, X_test

# Write and export to csv
def export_prediction(y_pred, file_name):
    prediction_df = Id
    prediction_df.insert(1, "Prediction", y_pred)
    print(prediction_df)
    prediction_df.to_csv('predictions/' + file_name, index=False)


if __name__ == "__main__":
    # Load data
    training_df = pd.read_csv('data/train.csv')
    testing_df = pd.read_csv('data/test.csv')

    # Save columns before dropping
    y_train = training_df.revenue.to_numpy()
    Id = testing_df[['Id']]

    # Baseline: Mean Revenue
    baseline_pred = baseline_model(training_df, testing_df)
    export_prediction(baseline_pred, 'baseline_prediction.csv')

    # Linear regression with city group
    simple_lin_reg_pred = simple_lin_reg_model(training_df, testing_df, y_train)
    export_prediction(simple_lin_reg_pred, 'simple-lin-reg-prediction.csv')

    # Pre-process data for the remaining models
    X_train, y_train, X_test = pre_process(training_df, testing_df)

    # Linear regression with L2 regularization
    lin_reg_l2_pred = lin_reg_l2_model(X_train, y_train, X_test)
    export_prediction(lin_reg_l2_pred, 'lin_reg_l2_prediction.csv')

    # Random forest regression
    rf_pred = rf_model(X_train, y_train, X_test)
    export_prediction(rf_pred, 'rf_prediction.csv')

    # 3-layer neural network
    nn_pred = nn_model(X_train, y_train, X_test)
    export_prediction(nn_pred, 'keras_nn_prediction.csv')
