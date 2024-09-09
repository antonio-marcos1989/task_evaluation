import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregue o dataset
file_path = 'dataset/tasks_with_sentiment.xlsx'  # Altere para o caminho correto do seu arquivo
data = pd.read_excel(file_path)

# Remova as colunas especificadas
columns_to_drop = ['id', 'fields_issuetype_id', 'fields_created', 'fields_description', 'fields_summary', 'fields_priority_id']
filtered_data = data.drop(columns=columns_to_drop)

# Remova as linhas com dados faltantes
cleaned_data = filtered_data.dropna()

# Defina X e y
X = cleaned_data.drop(columns=['status'])
y = cleaned_data['status']

# One-hot encoding para variáveis categóricas
X_encoded = pd.get_dummies(X)

# Divida os dados em conjuntos de treinamento e teste usando amostragem estratificada
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X_encoded, y))

X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Treine e avalie o modelo Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_f1_score = f1_score(y_test, y_pred_dt)

print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"Decision Tree F1-Score: {dt_f1_score:.2f}")

# Treine e avalie o modelo Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1_score = f1_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Random Forest F1-Score: {rf_f1_score:.2f}")

# Treine e avalie o modelo XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_f1_score = f1_score(y_test, y_pred_xgb)

print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")
print(f"XGBoost F1-Score: {xgb_f1_score:.2f}")

# Calcule a importância das características para o modelo XGBoost
feature_importances = xgb_model.feature_importances_
feature_names = X_train.columns

# Crie uma lista para exibir a importância das características
importances_list = list(zip(feature_names, feature_importances))

print("Importância das características (nome, importância):")
for feature, importance in importances_list:
    print(f"{feature}: {importance}")