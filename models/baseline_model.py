import numpy as np

# Uses the mean revenue from the training set
def baseline_model(training_df, testing_df):
    testing_df['Prediction'] = np.average(training_df['revenue'])
    prediction = testing_df.Prediction.to_numpy()
    return prediction
