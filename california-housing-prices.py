import os
import time

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


# Reading the the dataset and parsing it
csv_file = os.path.join("datasets", "california-housing-prices", "housing.csv")
df = pd.read_csv(csv_file)

# Converting the "ocean_proximity" column into 2 columns
if "ocean_proximity" in df.columns:
  df = pd.get_dummies(df, columns=["ocean_proximity"])

# Dropping missing values
df = df.dropna()

# Splitting the data
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42
)

# Configuring the random forest regressor
rf = RandomForestRegressor(
  n_estimators=100,
  max_features='sqrt',
  max_depth=None,
  min_samples_split=5,
  random_state=42,
  n_jobs=1
)

# Training the model
print(f"Starting the training process...")
start_time = time.time()
rf.fit(X_train, y_train)
end_time = time.time()
print(f"Completed the training in {(end_time - start_time):.2f}s")

# Calculating the predictions
y_pred = rf.predict(X_test)

# Calculating the stats
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the stats
print(f"\nStats:")
print(f"Mean squered error: {mse:.2f}")
print(f"Root mean squered error: {rmse:.2f}")
print(f"R^2 score: {r2:.2f}")

# Calculating the importance of each column
importances = rf.feature_importances_
features_names = X.columns
feat_imp_df = pd.DataFrame({
  "feature": features_names,
  "importance": importances
}).sort_values(by="importance", ascending=False)

# Printing the importance factors
print("\nFeature importance:")
print(feat_imp_df)