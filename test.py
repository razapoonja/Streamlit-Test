import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Churn_Modelling.csv")

# Features & Target
X = df[['Geography','Gender','Age','Balance','CreditScore',
        'Tenure','NumOfProducts','HasCrCard','IsActiveMember']]
y = df[['EstimatedSalary']]   # 🎯 target is Salary

# -----------------------------
# Preprocessing
# -----------------------------
categorical = ['Geography','Gender']
numeric = ['Age','Balance','CreditScore','Tenure','NumOfProducts','HasCrCard','IsActiveMember']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical),
    ('num', StandardScaler(), numeric)
])

# Fit & transform
X_processed = preprocessor.fit_transform(X)

# Save preprocessor (with pickle to avoid joblib mismatch)
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# -----------------------------
# ANN Model
# -----------------------------
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # regression output
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=32)

# Save Model (H5 format, no compile required for inference)
model.save("salary_model.h5")