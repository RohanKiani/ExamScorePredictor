import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("student_habits_performance.csv")

# Handle missing values
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])

# Drop unnecessary columns
df.drop(columns=['student_id'], inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
selected_features = [
    'study_hours_per_day',
    'exercise_frequency',
    'social_media_hours',
    'netflix_hours',
    'sleep_hours',
    'mental_health_rating',
    'attendance_percentage'
]
X = df[selected_features]
y = df['exam_score']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"✅ MAE: {mae:.2f}")
print(f"✅ MSE: {mse:.2f}")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R-squared: {r2:.2f}")

# Save model and scaler
joblib.dump(model, "linear_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
