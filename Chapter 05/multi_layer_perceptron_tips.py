import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the tips dataset from seaborn
tips_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')

# Preprocess the data
X = tips_df[['total_bill', 'size']]
y = tips_df['tip']

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create an MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=42)

# Train the MLP on the training data
mlp.fit(X_train, y_train)

# Make predictions on the test data
y_pred = mlp.predict(X_test)

# Calculate the root mean squared error (RMSE) of the MLP
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
