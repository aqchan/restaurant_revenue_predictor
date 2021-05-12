import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def rf_model(X_train, y_train, X_test):
    # Random Forest Regressor on all features using the optimal parameters
    # rf = RandomForestRegressor(n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',
    #                            max_depth=60, bootstrap=True).fit(X_train, y_train)

    rf = find_optimal_hyperparameters(X_train, y_train)
    print("Best hyperparameters:", rf.best_params_)

    y_pred = rf.predict(X_test)
    return y_pred


def find_optimal_hyperparameters(X_train, y_train):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42)

    # Fit the random search model
    rf_random.fit(X_train, y_train)
    return rf_random
