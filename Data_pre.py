import pandas as pd
import numpy as np
df = pd.read_csv('TCS.csv')
df['Volume'] = df['Volume'].astype(float)
df['Date'] = pd.to_datetime(df['Date'], format ='%Y-%m-%d')


stock_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
date_columns = ["Date"]

# Apply log transformation to the selected columns
df[stock_columns] = np.log1p(df[stock_columns] +1e-6)

# Define the features (X) and targets (y)
features = stock_columns # Stock columns and date columns as features
targets = ["Open", "Close"]  # "Open" and "Close" as targets

train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]
#print(train_data)
#print(test_data)