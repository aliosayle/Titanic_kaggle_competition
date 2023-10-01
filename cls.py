import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load CSV data into a DataFrame
df = pd.read_csv('train.csv')

# Data Preprocessing
# Select features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Handle missing values (e.g., filling missing ages with the mean)
X['Age'].fillna(X['Age'].mean(), inplace=True)

# Encode categorical variables (e.g., 'Sex' and 'Embarked')
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and Compile the Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

model.save('titanic.h5')
loaded_model = tf.keras.models.load_model('titanic.h5')

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
