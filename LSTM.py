import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.impute import SimpleImputer
import warnings
import csv

# Create a dictionary to store the LSTM results
lstm_results = {}
from Data_pre import train_data, test_data, features, targets

# Define different prediction horizons (n values)
prediction_horizons = [1, 3, 10, 30]

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the CSV file to store the results
csv_filename = 'lstm_results.csv'

# Open the CSV file in write mode
with open(csv_filename, 'w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Parameter', 'Prediction Horizon (n)', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    for param in targets:
        for n in prediction_horizons:
            # Split data into X (features) and y (target)
            X_train = train_data[features]
            y_train = train_data[param]  # Target is the current parameter
            X_test = test_data[features]
            y_test = test_data[param]  # Target is the current parameter

            # Create new target variables for each prediction horizon
            y_train_shifted = y_train.shift(-n)  # Shift target values n days into the future

            # Remove rows with NaN in the shifted target variable
            X_train = X_train[:-n]
            y_train_shifted = y_train_shifted.dropna()

            # Normalize the data for X_train
            scaler_X = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)

            # Normalize the data for X_test
            X_test_scaled = scaler_X.transform(X_test)

            # Normalize the target variable
            scaler_y = MinMaxScaler()
            y_train_shifted_scaled = scaler_y.fit_transform(np.array(y_train_shifted).reshape(-1, 1))

            # Create sequences for LSTM
            sequence_length = 30
            X_train_sequences = []
            y_train_sequences = []

            for i in range(sequence_length, len(X_train_scaled)):
                X_train_sequences.append(X_train_scaled[i - sequence_length:i, :])
                y_train_sequences.append(y_train_shifted_scaled[i, 0])

            X_train_sequences = np.array(X_train_sequences)
            y_train_sequences = np.array(y_train_sequences)

            # Reshape X_train_sequences to match LSTM input shape
            X_train_sequences = np.reshape(X_train_sequences, (
            X_train_sequences.shape[0], X_train_sequences.shape[1], X_train_sequences.shape[2]))

            # Create and train the LSTM model
            lstm_model = create_lstm_model((X_train_sequences.shape[1], X_train_sequences.shape[2]))
            lstm_model.fit(X_train_sequences, y_train_sequences, epochs=50, batch_size=32)

            # Prepare the test data for prediction
            X_test_sequences = []

            for i in range(sequence_length, len(X_test_scaled) - n):
                X_test_sequences.append(X_test_scaled[i - sequence_length:i, :])

            X_test_sequences = np.array(X_test_sequences)
            X_test_sequences = np.reshape(X_test_sequences, (
            X_test_sequences.shape[0], X_test_sequences.shape[1], X_test_sequences.shape[2]))

            # Make predictions for the test set
            lstm_predictions_scaled = lstm_model.predict(X_test_sequences)
            lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled)

            # Align the lengths of y_test and lstm_predictions
            y_test = y_test.iloc[sequence_length:-n].values
            lstm_predictions = lstm_predictions[:len(y_test)]  # Ensure same length

            # Remove rows with NaN values from both y_test and lstm_predictions
            nan_indices = np.isnan(y_test) | np.isnan(lstm_predictions)

            # Calculate and display the error metrics for the current parameter and prediction horizon
            nan_indices = np.isnan(y_test)
            y_test = y_test[~nan_indices].flatten()
            lstm_predictions = lstm_predictions[~nan_indices].flatten()

            mae = np.mean(np.abs(y_test - lstm_predictions))
            mse = np.mean((y_test - lstm_predictions) ** 2)
            rmse = np.sqrt(mse)


            def calculate_mape(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


            mape = calculate_mape(y_test, lstm_predictions)

            # Display metrics for the current parameter and prediction horizon
            print(f"Parameter: {param}, Prediction Horizon (n): {n} days")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
            print()
            lstm_results[(param, n)] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            }
            # Append the results to the CSV file
            csv_writer.writerow([param, n, mae, mse, rmse, mape])

 # Save the results in a CSV file
print(f"LSTM Results saved to {csv_filename}")

