import pandas as pd

# Load results from the RF and LSTM CSV files
rf_results_df = pd.read_csv('rf_results.csv')
lstm_results_df = pd.read_csv('lstm_results.csv')

# Define the weights for RF and LSTM models
rf_weight = 0.5  # Weight for RF predictions
lstm_weight = 0.5  # Weight for LSTM predictions

# Define the parameters and prediction horizons
parameters = ['Open', 'Close']
prediction_horizons = [1, 3, 10, 30]

# Create a list to store DataFrames
dataframes = []

# Iterate over parameters and prediction horizons
for param in parameters:
    for horizon in prediction_horizons:
        # Get RF and LSTM results for the current parameter and horizon
        rf_row = rf_results_df[(rf_results_df['Parameter'] == param) & (rf_results_df['Prediction Horizon (n)'] == horizon)].iloc[0]
        lstm_row = lstm_results_df[(lstm_results_df['Parameter'] == param) & (lstm_results_df['Prediction Horizon (n)'] == horizon)].iloc[0]

        # Calculate the weighted average for each evaluation metric
        weighted_avg_results = {}
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            weighted_avg = rf_weight * rf_row[metric] + lstm_weight * lstm_row[metric]
            weighted_avg_results[metric] = round(weighted_avg, 4)

        # Create a DataFrame for the current result
        df = pd.DataFrame({
            'Parameter': [param],
            'Prediction Horizon (n)': [horizon],
            'MAE': [weighted_avg_results['MAE']],
            'MSE': [weighted_avg_results['MSE']],
            'RMSE': [weighted_avg_results['RMSE']],
            'MAPE': [weighted_avg_results['MAPE']]
        })

        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate the DataFrames
average_results_df = pd.concat(dataframes, ignore_index=True)

# Save the average results to a CSV file
average_results_df.to_csv('rf_lstm.csv', index=False)
print("Average RF-LSTM Results saved to rf_lstm.csv")
