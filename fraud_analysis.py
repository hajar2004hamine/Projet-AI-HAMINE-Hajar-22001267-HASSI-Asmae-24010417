import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import json
import os

# 1. GENERATION DE DONNEES SIMULEES
np.random.seed(42)
n_samples = 5000

# Features:
montant = np.random.lognormal(mean=np.log(100), sigma=1.2, size=n_samples)
depense_weekend = np.random.binomial(1, 0.2, size=n_samples)
distance = np.random.exponential(scale=50, size=n_samples)
historique = np.random.poisson(lam=0.5, size=n_samples)
dpt_mean_ratio = np.random.normal(loc=1.0, scale=0.3, size=n_samples)

prob_fraude = (
    0.3 * (montant > 500) * depense_weekend + 
    0.4 * (distance > 200) * (historique > 2) +
    0.2 * (dpt_mean_ratio > 3) +
    np.random.uniform(0, 0.05, n_samples)
)
is_fraud = (prob_fraude > 0.4).astype(int)

# Seuils d'alerte automatiques (feature dérivée)
depassement_seuil = ((montant > 1000) | (dpt_mean_ratio > 2.5)).astype(int)

df = pd.DataFrame({
    'montant': montant,
    'depense_weekend': depense_weekend,
    'distance_km': distance,
    'historique_anomalies': historique,
    'ratio_moyenne_departement': dpt_mean_ratio,
    'alerte_declenchee': depassement_seuil,
    'is_fraud': is_fraud
})

print(f"Jeu de données généré : {len(df)} lignes, {df['is_fraud'].sum()} fraudes ({(df['is_fraud'].sum()/len(df))*100:.2f}%).")

# PREPROCESSING
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

metrics_dict = {}

def evaluate_model(y_true, y_pred, y_prob, model_name):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None
    }
    metrics_dict[model_name] = metrics
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matrice de Confusion\n{model_name}')
    plt.ylabel('Vrai Label')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. LOGISTIC REGRESSION
print("Entrainement Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

evaluate_model(y_test, y_pred_lr, y_prob_lr, 'Logistic_Regression')
plot_confusion_matrix(y_test, y_pred_lr, 'Régression Logistique', 'cm_lr.png')

# 2. RANDOM FOREST
print("Entrainement Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

evaluate_model(y_test, y_pred_rf, y_prob_rf, 'Random_Forest')
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest', 'cm_rf.png')

fi = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x='importance', y='feature', data=fi, hue='feature', dodge=False, palette='viridis', legend=False)
plt.title('Importance des variables (Random Forest)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# 3. ISOLATION FOREST
print("Entrainement Isolation Forest...")
iso = IsolationForest(contamination=df['is_fraud'].sum()/len(df), random_state=42)
iso.fit(X_train_scaled)
y_pred_iso_raw = iso.predict(X_test_scaled)
y_pred_iso = np.where(y_pred_iso_raw == -1, 1, 0)
y_prob_iso = -iso.score_samples(X_test_scaled)

evaluate_model(y_test, y_pred_iso, y_prob_iso, 'Isolation_Forest')
plot_confusion_matrix(y_test, y_pred_iso, 'Isolation Forest', 'cm_iso.png')

# ROC CURVES
plt.figure(figsize=(8,6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC = {metrics_dict['Logistic_Regression']['roc_auc']:.3f})")

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {metrics_dict['Random_Forest']['roc_auc']:.3f})")

fpr_iso, tpr_iso, _ = roc_curve(y_test, y_prob_iso)
plt.plot(fpr_iso, tpr_iso, label=f"IsoForest (AUC = {metrics_dict['Isolation_Forest']['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title('Courbes ROC')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("Analyse terminée ! Résultats sauvegardés.")
