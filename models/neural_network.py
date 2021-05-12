from tensorflow import keras
from sklearn.model_selection import KFold

# Deep model: 3-layer neural network with K-fold cross validation
def nn_model(X_train, y_train, X_test):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(X_train, y_train):
        # Neural Network
        model = keras.Sequential()
        model.add(keras.layers.Dense(30, input_dim=43, activation="relu"))  # layer 1 (hidden)
        model.add(keras.layers.Dense(12, activation="relu"))  # layer 2 (hidden)
        model.add(keras.layers.Dense(1, activation="linear"))  # layer 3 (output)

        optimizer = keras.optimizers.Adam(lr=0.045)
        model.compile(optimizer=optimizer, loss='mse')

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        model.fit(X_train[train], y_train[train], batch_size=32, epochs=100, validation_split = 0.2, verbose=1)

        # Increase fold number
        fold_no = fold_no + 1

    y_pred = model.predict(X_test)
    return y_pred
