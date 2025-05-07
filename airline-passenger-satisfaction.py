import os
import time

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Reading the datasets
train_csv_file = os.path.join("datasets", "airline-passenger-satisfaction", "train.csv")
test_csv_file = os.path.join("datasets", "airline-passenger-satisfaction", "test.csv")

# Parsing the datasets
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

# Dropping the etc columns
def drop_misc_columns(df: pd.DataFrame) -> pd.DataFrame:
  if "id" in df.columns:
    df = df.drop("id", axis=1)

  if "Gender" in df.columns:
    df = df.drop("Gender", axis=1)

  if df.columns[0].title() != "Customer Type":
    df = df.drop(df.columns[0], axis=1)

  return df


# Mapping values into numeric values
def map_obj_columns(df: pd.DataFrame) -> pd.DataFrame:
  satisfaction_map = {
    "neutral or dissatisfied": 0,
    "satisfied": 1
  }

  customer_type_map = {
    "disloyal Customer": 0,
    "Loyal Customer": 1
  }

  flight_class_map = {
    "Eco": 0,
    "Eco Plus": 1,
    "Business": 2
  }

  df["satisfaction"] = df["satisfaction"].map(satisfaction_map)
  df["Customer Type"] = df["Customer Type"].map(customer_type_map)
  df["Class"] = df["Class"].map(flight_class_map)

  if "Type of Travel" in df.columns:
    df = pd.get_dummies(df, columns=["Type of Travel"])

  return df


# Processing the datasets
train_df = drop_misc_columns(train_df)
train_df = map_obj_columns(train_df)

test_df = drop_misc_columns(test_df)
test_df = map_obj_columns(test_df)

# Splitting the datasets
X_train = train_df.drop("satisfaction", axis=1)
y_train = train_df["satisfaction"]
X_test = test_df.drop("satisfaction", axis=1)
y_test = test_df["satisfaction"]

# Configuring the random forest regressor
rf = RandomForestClassifier(
  n_estimators=100,
  max_features='sqrt',
  max_depth=None,
  min_samples_split=25,
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
pred_start = time.perf_counter_ns()
y_pred = rf.predict(X_test)
pred_end = time.perf_counter_ns()

pred_time = (pred_end - pred_start) / 1000
average_pred_time = pred_time / X_test.shape[0]

# Calculating the stats
report = classification_report(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

# Printing the stats
print(f"\nAccuracy: {score * 100:.2f}%")
print(f"Prediction time per sample {average_pred_time:.2f} microseconds")
print("Classification Report:")
print(report)

# Calculating the importance of each column
features_names = X_test.columns
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
  "feature": features_names,
  "importance": importances
}).sort_values(by="importance", ascending=False)

# Printing the importance factors
print("\nFeature importance:")
print(feat_imp_df)