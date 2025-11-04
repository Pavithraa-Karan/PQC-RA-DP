import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# Load dataset
df = pd.read_csv("qars_scored.csv")
df

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Features and target
X = df.drop(['QARS_category', 'QARS', 'q_derived'], axis=1)
y = df['QARS_category']


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model training with Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoders['QARS_category'].classes_,
            yticklabels=label_encoders['QARS_category'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Differentially Private Logistic Regression
epsilons = [0.1, 0.5, 1, 2, 5, 10]
dp_results = []
trials = 10
for eps in epsilons:
    acc_list = []
    for _ in range(trials):
        dp_lr = DPLogisticRegression(epsilon=eps, data_norm=6.0, random_state=42)
        dp_lr.fit(X_train, y_train)
        y_pred = dp_lr.predict(X_test)
        acc_list.append(accuracy_score(y_test, y_pred))
    avg_acc = np.mean(acc_list)
    dp_results.append((eps, avg_acc))

# Create DataFrame for results
dp_df = pd.DataFrame(dp_results, columns=['Epsilon', 'Accuracy'])
print("\nDP Trade-Off Table:\n")
print(dp_df)



# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoders['QARS_category'].classes_,
            yticklabels=label_encoders['QARS_category'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Differential Privacy')
plt.show()


from sklearn.linear_model import SGDClassifier
import joblib

# --- Adaptive Learning with SGDClassifier (Logistic Regression behavior) ---
adaptive_model = SGDClassifier(
    loss='log_loss',           # logistic regression equivalent
    learning_rate='optimal',
    eta0=0.01,
    max_iter=1000,
    random_state=42
)

# Initial training
adaptive_model.fit(X_train, y_train)
print("\nInitial Adaptive Model Accuracy:", accuracy_score(y_test, adaptive_model.predict(X_test)))

# --- Simulate incoming new data ---
# Here we simulate feedback or new incoming samples (could be real in production)
new_samples = df.sample(1, random_state=99)  # example: one new row
X_new = scaler.transform(new_samples.drop(['QARS_category', 'QARS', 'q_derived'], axis=1))
y_new = new_samples['QARS_category']

# --- Add Gaussian noise to model weights (manual DP-like update) ---
# This helps ensure privacy by slightly perturbing learned weights
def add_gaussian_noise_to_weights(model, sigma=0.05):
    for i in range(len(model.coef_)):
        noise = np.random.normal(0, sigma, model.coef_[i].shape)
        model.coef_[i] += noise
    model.intercept_ += np.random.normal(0, sigma, model.intercept_.shape)
    return model

# --- Adaptive update (partial_fit + DP noise) ---
adaptive_model.partial_fit(X_new, y_new, classes=np.unique(y_train))
adaptive_model = add_gaussian_noise_to_weights(adaptive_model, sigma=0.05)

# --- Evaluate after update ---
y_pred_adaptive = adaptive_model.predict(X_test)
print("Post-Update Adaptive Model Accuracy:", accuracy_score(y_test, y_pred_adaptive))

# Optional: Save model state to reuse in next session
joblib.dump(adaptive_model, "adaptive_dp_model.pkl")
print("Adaptive DP model saved as 'adaptive_dp_model.pkl'")