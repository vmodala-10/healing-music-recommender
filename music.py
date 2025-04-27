import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("healing_music_dataset.csv")

# Encode categorical features
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split dataset
X = df.drop("Raga", axis=1)
y = df["Raga"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Individual models
dt = DecisionTreeClassifier(random_state=1)
knn = KNeighborsClassifier()
rf = RandomForestClassifier(n_estimators=100, random_state=1)

# Ensemble: Voting Classifier
ensemble = VotingClassifier(estimators=[
    ('dt', dt),
    ('knn', knn),
    ('rf', rf)
], voting='hard')  # You can try 'soft' if models support predict_proba

# Train
ensemble.fit(X_train, y_train)

# Predict
y_pred = ensemble.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Ensemble Model Accuracy: {acc*100:.2f} %")

# Save the model and encoders
joblib.dump(ensemble, 'ensemble_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')
print("📦 Ensemble model and encoders saved.")
