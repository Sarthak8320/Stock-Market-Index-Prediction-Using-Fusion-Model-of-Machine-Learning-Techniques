from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer
import csv
from Data_pre import train_data, test_data,targets,features

# Rest of your code
# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=150,     # Number of trees
    max_depth=5,       # Maximum depth of trees
    min_samples_split=8,  # Minimum samples required to split
    min_samples_leaf=20,   # Minimum samples required in a leaf
    max_features=3,  # Number of features to consider for split
    bootstrap=True,       # Use bootstrapping
    random_state=50       # Random seed for reproducibility
)

# Define the CSV file to store the results
csv_filename = 'rf_results.csv'
prediction_horizons = [1, 3, 10, 30]
# Open the CSV file in write mode
with open(csv_filename, 'w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Parameter', 'Prediction Horizon (n)', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    for param in targets:
        for n in prediction_horizons:
            # Your existing code for calculations
            # Split data into X (features) and y (target)
            X_train = train_data[features]
            y_train = train_data[param]  # Target is the current parameter
            X_test = test_data[features]
            y_test = test_data[param]  # Target is the current parameter

            # Feature Engineering:
            # Include lag features
            for lag in range(1, 6):
                X_train[f'Lag{lag}'] = y_train.shift(lag)
                X_test[f'Lag{lag}'] = y_test.shift(lag)

            # Calculate 10-day simple moving average
            X_train['SMA_10'] = y_train.rolling(window=10).mean()
            X_test['SMA_10'] = y_test.rolling(window=10).mean()

            # Create new target variables for each prediction horizon
            y_train_shifted = y_train.shift(-n)  # Shift target values n days into the future

            # Remove rows with NaN in the shifted target variable
            X_train = X_train[:-n]
            y_train_shifted = y_train_shifted.dropna()

            # Create a SimpleImputer instance with the desired strategy
            imputer = SimpleImputer(strategy='mean')

            # Fit the imputer on the training data and transform both training and test data
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            # Train the Random Forest model using the imputed data
            rf_model.fit(X_train_imputed, y_train_shifted)

            # Make predictions for the test set
            rf_predictions = rf_model.predict(X_test_imputed[:-n])
            # Calculate and display the error metrics for the current parameter and prediction horizon
            mae = mean_absolute_error(y_test.iloc[:-n], rf_predictions)
            mse = mean_squared_error(y_test.iloc[:-n], rf_predictions)
            rmse = np.sqrt(mse)


            def calculate_mape(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


            mape = calculate_mape(y_test.iloc[:-n], rf_predictions)

            # Display metrics for the current parameter and prediction horizon
            print(f"Parameter: {param}, Prediction Horizon (n): {n} days")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
            print()

            # Append the results to the CSV file
            csv_writer.writerow([param, n, mae, mse, rmse, mape])

# Save the results in a CSV file
print(f"Results saved to {csv_filename}")
