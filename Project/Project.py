import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load the data
file_path = r"E:\EDGE\1970-2023 Oil spill .csv"  # Path to the uploaded CSV file
data = pd.read_csv(file_path)

# Display the dataset columns
print("Dataset Columns:", data.columns)

# Ensure the required columns exist
required_columns = ['Year', 'Quantity of Oil Spilled']
if all(column in data.columns for column in required_columns):
    # Extract only relevant columns
    data = data[required_columns]

    # Handle missing values
    print("\nHandling missing values...")
    missing_values_before = data.isnull().sum()
    print("Missing values before handling:\n", missing_values_before)

    imputer = SimpleImputer(strategy='mean')
    data[['Quantity of Oil Spilled']] = imputer.fit_transform(data[['Quantity of Oil Spilled']])

    missing_values_after = data.isnull().sum()
    print("Missing values after handling:\n", missing_values_after)

    # Extract features (Year) and target (Quantity of Oil Spilled)
    X = data[['Year']].values  # Feature: Year
    y = data['Quantity of Oil Spilled'].values  # Target: Quantity of Oil Spilled

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Fitting the training data using Linear Regression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting from the learned model
    y_predict = regressor.predict(X_test)

    print("\nOriginal values (Test set):", y_test)
    print("\nPredicted values:", y_predict)

    # Visualize the data
    plt.scatter(X_train, y_train, color='green', label='Training Data')
    plt.plot(X_train, regressor.predict(X_train), color='red', label='Regression Line')
    plt.scatter(X_test, y_test, color='blue', label='Test Data')
    plt.title('Year vs Quantity of Oil Spilled')
    plt.xlabel('Year')
    plt.ylabel('Quantity of Oil Spilled')
    plt.legend()
    plt.show()

    # Predict oil spilling for future years
    future_years = np.array(range(2024, 2035)).reshape(-1, 1)
    future_predictions = regressor.predict(future_years)

    print("\nPredictions for future years (2024-2034):")
    for year, prediction in zip(future_years.flatten(), future_predictions):
        print(f"Year: {year}, Predicted Quantity of Oil Spilled: {prediction:.2f}")

    # Visualize predictions for future years
    plt.scatter(future_years, future_predictions, color='purple', label='Future Predictions')
    plt.plot(future_years, future_predictions, color='orange', linestyle='--', label='Future Trend')
    plt.title('Predicted Future Oil Spilling')
    plt.xlabel('Year')
    plt.ylabel('Quantity of Oil Spilled')
    plt.legend()
    plt.show()
else:
    print("Error: Required columns 'Year' and 'Quantity of Oil Spilled' are missing from the dataset.")
