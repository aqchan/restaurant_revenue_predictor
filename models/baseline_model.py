import numpy as np

# Uses the mean revenue from the training set
def baseline_model(training_df, testing_df):
    average_revenue = np.average(training_df['revenue'])
    prediction = np.ones(testing_df.shape[0]) * average_revenue
    return prediction
