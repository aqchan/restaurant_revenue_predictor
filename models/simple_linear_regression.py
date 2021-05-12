from sklearn.linear_model import LinearRegression

def simple_lin_reg_model(training_df, testing_df, y_train):
    # Convert all columns with a space to an underscore
    training_df.columns = training_df.columns.str.replace(' ', '_')
    testing_df.columns = testing_df.columns.str.replace(' ', '_')

    # Convert city group to weights, where Big Cities = 1 and Other = 0
    X_train = training_df.City_Group.map({'Big Cities': 1, 'Other': 0}).to_numpy()
    X_test = testing_df.City_Group.map({'Big Cities': 1, 'Other': 0}).to_numpy()

    # Reshape features
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Run regression model
    reg = LinearRegression().fit(X_train, y_train)

    # Compute predictions on test set
    y_pred = reg.predict(X_test)
    return y_pred
