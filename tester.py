import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model = tf.keras.models.load_model("titanic.h5")

# Load the training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine both training and test data for encoding (to ensure consistent labels)
combined_df = pd.concat([train_df, test_df], axis=0)

# Define Label Encoders for Categorical Columns
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    label_encoders[col] = le

# Separate the combined data back into training and test data
train_len = len(train_df)
X_train = combined_df[:train_len]
X_test = combined_df[train_len:]

# Data Preprocessing for Test Data
# Handle missing values
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)

# Make Predictions
predictions = model.predict(X_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])

# Convert predicted probabilities to binary labels (0 or 1)
predicted_labels = (predictions > 0.5).astype(int)

# Create a DataFrame with PassengerId and Survived columns
result_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predicted_labels.flatten()})

# Save the result to a CSV file
result_df.to_csv('result.csv', index=False)
