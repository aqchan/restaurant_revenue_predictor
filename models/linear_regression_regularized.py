from sklearn.linear_model import Ridge

# Linear Regression with L2 Regularization
def lin_reg_l2_model(X_train, y_train, X_test):
    lr = Ridge(alpha=300.0).fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return y_pred
