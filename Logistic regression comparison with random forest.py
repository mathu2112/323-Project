import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


# 1. DATA PREPROCESSING

df = pd.read_csv("/content/MiddleEast_dataset.csv")

df = df.drop_duplicates()
# Remove missing Emirate_Region
df = df.dropna(subset=["Emirate"])
df["Budget"] = df["Budget"].fillna(df["Budget"].median())
df["Labor_Hours"] = df["Labor_Hours"].fillna(df["Labor_Hours"].median())

# Convert Dates
df["Start_Date"] = pd.to_datetime(df["Start_Date"], dayfirst=True)
df["End_Date"] = pd.to_datetime(df["End_Date"], dayfirst=True)
df["Actual_End_Date"] = pd.to_datetime(df["Actual_End_Date"], dayfirst=True)

# Create Target Variable
df['On_Time_Status'] = np.where(
    df['Actual_End_Date'] > df['End_Date'],
    1,
    0
)
print("Target Variable Distribution:")
print(df['On_Time_Status'].value_counts())
print("-" * 50)

# Planned Duration
df['Planned_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days
df['Start_Month'] = df['Start_Date'].dt.month
df['Start_DayOfWeek'] = df['Start_Date'].dt.dayofweek

df = df.drop(columns=[
    'Project_ID',
    'Project_Name',
    'Actual_End_Date',
    'Task_Status',
    'Cost',
    'Cost_Variance',
    'Start_Date',
    'End_Date'
])

# Encoding text to numbers
le_emirate = LabelEncoder()
le_work = LabelEncoder()
le_contractor = LabelEncoder()

df['Emirate'] = le_emirate.fit_transform(df['Emirate'])
df['Type_of_Work'] = le_work.fit_transform(df['Type_of_Work'])
df['Contractor'] = le_contractor.fit_transform(df['Contractor'])

X = df.drop('On_Time_Status', axis=1)
y = df['On_Time_Status']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Logistic Regression & SVM require feature scaling to converge properly.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. RANDOM FOREST (With GridSearch)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from scipy import stats



# ====================================
# 3. Model Initialization
# ====================================

log_reg = LogisticRegression(random_state=42)

rf = best_rf

# ====================================
# 4. Statistical Comparison
# ====================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

log_reg_scores = cross_val_score(
    log_reg,
    scaler.fit_transform(X),
    y,
    cv=cv,
    scoring='accuracy'
)

rf_scores = cross_val_score(
    best_rf,
    X,
    y,
    cv=cv,
    scoring='accuracy'
)

t_stat, p_value = stats.ttest_rel(
    log_reg_scores,
    rf_scores
)

# ====================================
# 5. Model Training
# ====================================

log_reg.fit(X_train_scaled, y_train)
best_rf.fit(X_train, y_train)

# ====================================
# 6. Predictions
# ====================================

y_pred_log = log_reg.predict(X_test_scaled)
y_pred_rf = best_rf.predict(X_test)

# ====================================
# 7. Accuracy
# ====================================

print("----- Logistic Regression -----")
print("Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))


print("\n\n----- Random Forest -----")
print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# ====================================
# 8. ROC Curve
# ====================================

y_prob_log = log_reg.predict_proba(X_test_scaled)[:,1]
y_prob_rf = best_rf.predict_proba(X_test)[:,1]

fpr_log, tpr_log, _ = roc_curve(
    y_test,
    y_prob_log
)

# fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)

auc_log = auc(fpr_log, tpr_log)
# auc_rf = auc(fpr_rf, tpr_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(
    fpr_log,
    tpr_log,
    label=f'Logistic Regression (AUC = {auc_log:.3f})'
)

plt.plot(
    fpr,
    tpr,
    label=f'Random Forest (AUC = {roc_auc:.3f})'
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()

plt.show()

# ====================================
# 9. Confusion Matrices
# ====================================

cm_log = confusion_matrix(
    y_test,
    y_pred_log
)

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))

plt.subplot(1,2,1)

sns.heatmap(
    cm_log,
    annot=True,
    fmt='d',
    cmap='Greens',
    cbar=False,
    xticklabels=['On Time','Delayed'],
    yticklabels=['On Time','Delayed']
)

plt.title('Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1,2,2)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['On Time','Delayed'],
    yticklabels=['On Time','Delayed']
)

plt.title('Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# ====================================
# 10. Statistical Results
# ====================================

print("\n----- Statistical Comparison -----")

print(
    "Logistic Regression Avg Accuracy:",
    np.mean(log_reg_scores)
)

print(
    "Random Forest Avg Accuracy:",
    np.mean(rf_scores)
)

print(
    "Paired T-Test p-value:",
    p_value
)

if p_value < 0.05:
    print(
        "Conclusion: Difference is statistically significant."
    )
else:
    print(
        "Conclusion: No statistically significant difference."
    )

print("\n----- AUC score -----")
print(f"Logistic Regression AUC Score: {auc_log:.4f}")
print(f"Random Forest AUC Score: {roc_auc:.4f}")
