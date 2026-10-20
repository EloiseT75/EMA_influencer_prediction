import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CATBOOST TRAINING V3 - HYPEROPTIMIZED")
print("="*60)

# =========================
# CHARGEMENT
# =========================
print("\n Chargement des features...")
X_train = pd.read_csv('X_train_features.csv')
X_test = pd.read_csv('X_test_features.csv')
y_train = pd.read_csv('y_train.csv')['label'].values
test_ids = pd.read_csv('test_ids.csv')['challenge_id'].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Distribution des labels: {np.bincount(y_train)}")
print(f"Classe 0: {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%)")
print(f"Classe 1: {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%)")

# =========================
# HYPERPARAMÈTRES OPTIMISÉS V3
# =========================
print("\n Configuration des hyperparamètres...")

params = {
    # Iterations et learning
    'iterations': 5000,           # Augmenté pour convergence
    'learning_rate': 0.008,       # Réduit pour meilleure généralisation
    
    # Structure de l'arbre
    'depth': 9,                   # Augmenté pour capturer interactions complexes
    'l2_leaf_reg': 5,             # Régularisation accrue
    'border_count': 254,
    'min_data_in_leaf': 20,       # Évite l'overfitting sur petits groupes
    
    # Randomisation et bagging
    'bagging_temperature': 1.0,   # Plus de bagging
    'random_strength': 1.5,       # Plus de randomisation
    'subsample': 0.8,             # Échantillonnage des données
    'colsample_bylevel': 0.8,     # Sélection aléatoire de features
    
    # Early stopping
    'od_type': 'Iter',
    'od_wait': 150,               # Plus patient
    
    # Autres
    'random_seed': 42,
    'verbose': 250,
    'task_type': 'CPU',
    'thread_count': -1,
    'eval_metric': 'Accuracy',
    
    # Balance des classes si nécessaire
    'auto_class_weights': 'Balanced' if abs((y_train==0).mean() - 0.5) > 0.1 else None
}

print("Paramètres configurés:")
for key, value in params.items():
    if value is not None:
        print(f"  {key}: {value}")

# =========================
# CROSS-VALIDATION STRATIFIÉE
# =========================
print("\n Cross-validation stratifiée (5-fold)...")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
oof_probs = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n{'='*60}")
    print(f" FOLD {fold}/{n_splits}")
    print(f"{'='*60}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"Train distribution: {np.bincount(y_tr)}")
    print(f"Val distribution: {np.bincount(y_val)}")
    
    # Pools CatBoost
    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)
    
    # Modèle
    model = CatBoostClassifier(**params)
    
    # Entraînement
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose_eval=250,
        plot=False
    )
    
    # Prédictions
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)
    
    oof_preds[val_idx] = val_preds
    oof_probs[val_idx] = val_probs
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    
    # Métriques
    fold_acc = accuracy_score(y_val, val_preds)
    cv_scores.append(fold_acc)
    
    print(f"\n Fold {fold} Results:")
    print(f"  Accuracy: {fold_acc:.4f}")
    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Best score: {model.best_score_}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    
    # Sauvegarder le modèle
    models.append(model)

# =========================
# RÉSULTATS GLOBAUX
# =========================
print(f"\n{'='*60}")
print(f" RÉSULTATS GLOBAUX DE LA CROSS-VALIDATION")
print(f"{'='*60}")

oof_accuracy = accuracy_score(y_train, oof_preds)
print(f"\n Out-of-Fold (OOF) Accuracy: {oof_accuracy:.4f}")
print(f" CV Scores par fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f" Mean CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Confusion matrix globale
cm_oof = confusion_matrix(y_train, oof_preds)
print(f"\n Confusion Matrix OOF:")
print(f"  True Negative (TN): {cm_oof[0, 0]}")
print(f"  False Positive (FP): {cm_oof[0, 1]}")
print(f"  False Negative (FN): {cm_oof[1, 0]}")
print(f"  True Positive (TP): {cm_oof[1, 1]}")

print(f"\n Classification Report:")
print(classification_report(y_train, oof_preds, target_names=['Observer', 'Influencer']))

# Sauvegarder les prédictions OOF
pd.DataFrame({
    'oof_preds': oof_preds,
    'oof_probs': oof_probs,
    'true_label': y_train
}).to_csv('catboost_oof_predictions.csv', index=False)
print("\n OOF predictions sauvegardées: catboost_oof_predictions.csv")

# =========================
# ANALYSE DES FEATURES
# =========================
print(f"\n{'='*60}")
print(f" ANALYSE DES FEATURES")
print(f"{'='*60}")

# Feature importance moyenne sur tous les folds
feature_importance = np.zeros(X_train.shape[1])
for model in models:
    feature_importance += model.feature_importances_ / n_splits

feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n Top 30 features les plus importantes:")
print(feature_importance_df.head(30).to_string(index=False))

# Sauvegarder l'importance des features
feature_importance_df.to_csv('feature_importance.csv', index=False)
print("\n Feature importance sauvegardée: feature_importance.csv")

# =========================
# ENTRAÎNEMENT FINAL SUR TOUT LE DATASET
# =========================
print(f"\n{'='*60}")
print(f" ENTRAÎNEMENT FINAL SUR LE DATASET COMPLET")
print(f"{'='*60}")

# Utiliser le nombre optimal d'itérations
best_iterations = [model.best_iteration_ for model in models]
optimal_iterations = int(np.mean(best_iterations))
print(f"\n Itérations moyennes optimales: {optimal_iterations}")
print(f"   Range: {min(best_iterations)} - {max(best_iterations)}")

# Ajuster légèrement à la hausse pour le full dataset
final_iterations = optimal_iterations + 50
print(f"   Itérations finales: {final_iterations}")

# Configuration finale
final_params = params.copy()
final_params['iterations'] = final_iterations
final_params['verbose'] = 250

# Entraînement final
final_pool = Pool(X_train, y_train)
final_model = CatBoostClassifier(**final_params)

print("\n Entraînement en cours...")
final_model.fit(final_pool, verbose_eval=250)

# Prédictions finales
print("\n Génération des prédictions finales...")
final_test_probs = final_model.predict_proba(X_test)[:, 1]
final_test_preds = (final_test_probs > 0.5).astype(int)

print(f"\n Distribution des prédictions test:")
print(f"  Classe 0 (Observer): {(final_test_preds==0).sum()} ({(final_test_preds==0).mean()*100:.1f}%)")
print(f"  Classe 1 (Influencer): {(final_test_preds==1).sum()} ({(final_test_preds==1).mean()*100:.1f}%)")

# =========================
# SAUVEGARDE DES MODÈLES
# =========================
print(f"\n{'='*60}")
print(f" SAUVEGARDE DES MODÈLES ET RÉSULTATS")
print(f"{'='*60}")

# Sauvegarder le modèle final
final_model.save_model('catboost_model_final.cbm')
print(" Modèle CatBoost: catboost_model_final.cbm")

with open('catboost_model_final.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print(" Modèle Pickle: catboost_model_final.pkl")

# Sauvegarder tous les modèles de CV
with open('catboost_cv_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print(" Modèles CV: catboost_cv_models.pkl")

# =========================
# GÉNÉRATION DE LA SOUMISSION
# =========================
print(f"\n{'='*60}")
print(f" GÉNÉRATION DU FICHIER DE SOUMISSION")
print(f"{'='*60}")

# Soumission finale
submission = pd.DataFrame({
    'ID': test_ids,
    'Prediction': final_test_preds
})
submission.to_csv('catboost_submission_final.csv', index=False)
print("\n Soumission finale: catboost_submission_final.csv")

# Soumission avec probabilités (pour analyse)
submission_probs = pd.DataFrame({
    'ID': test_ids,
    'Prediction': final_test_preds,
    'Probability': final_test_probs
})
submission_probs.to_csv('catboost_submission_with_probs.csv', index=False)
print(" Soumission avec probas: catboost_submission_with_probs.csv")

# =========================
# OPTIMISATION DU SEUIL (OPTIONNEL)
# =========================
print(f"\n{'='*60}")
print(f" OPTIMISATION DU SEUIL DE DÉCISION")
print(f"{'='*60}")

# Tester différents seuils sur les OOF predictions
thresholds = np.arange(0.3, 0.7, 0.05)
threshold_scores = []

for threshold in thresholds:
    preds_at_threshold = (oof_probs > threshold).astype(int)
    acc = accuracy_score(y_train, preds_at_threshold)
    threshold_scores.append(acc)
    print(f"Seuil {threshold:.2f}: Accuracy = {acc:.4f}")

best_threshold_idx = np.argmax(threshold_scores)
best_threshold = thresholds[best_threshold_idx]
best_threshold_acc = threshold_scores[best_threshold_idx]

print(f"\n Meilleur seuil: {best_threshold:.2f} (Accuracy: {best_threshold_acc:.4f})")

if best_threshold != 0.5 and best_threshold_acc > oof_accuracy + 0.001:
    print(f"   Amélioration de {(best_threshold_acc - oof_accuracy)*100:.2f}% avec seuil {best_threshold:.2f}")
    print(f"   Considérez d'utiliser ce seuil pour la soumission finale")
    
    # Générer une soumission alternative avec le meilleur seuil
    optimized_preds = (final_test_probs > best_threshold).astype(int)
    submission_optimized = pd.DataFrame({
        'ID': test_ids,
        'Prediction': optimized_preds
    })
    submission_optimized.to_csv('catboost_submission_optimized_threshold.csv', index=False)
    print(f"\n Soumission avec seuil optimisé: catboost_submission_optimized_threshold.csv")
    print(f"   Distribution: Classe 0: {(optimized_preds==0).sum()}, Classe 1: {(optimized_preds==1).sum()}")
else:
    print(f"   Le seuil par défaut (0.5) est optimal")

# =========================
# RÉSUMÉ FINAL
# =========================
print(f"\n{'='*60}")
print(f" RÉSUMÉ FINAL")
print(f"{'='*60}")
print(f"\n Performances:")
print(f"   OOF Accuracy: {oof_accuracy:.4f}")
print(f"   CV Mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"   Meilleur fold: {max(cv_scores):.4f}")
print(f"   Pire fold: {min(cv_scores):.4f}")

print(f"\n Configuration:")
print(f"   Features: {X_train.shape[1]}")
print(f"   Depth: {params['depth']}")
print(f"   Learning rate: {params['learning_rate']}")
print(f"   Iterations finales: {final_iterations}")

print(f"\n Fichiers générés:")
print(f"   catboost_submission_final.csv (soumission principale)")
print(f"   catboost_model_final.cbm (modèle entraîné)")
print(f"   catboost_oof_predictions.csv (prédictions OOF)")
print(f"   feature_importance.csv (importance des features)")
if best_threshold != 0.5 and best_threshold_acc > oof_accuracy + 0.001:
    print(f"   catboost_submission_optimized_threshold.csv (seuil optimisé)")

print(f"\n{'='*60}")
print(f" ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print(f"{'='*60}")
print(f"\n Prochaines étapes:")
print(f"   1. Vérifier les résultats dans catboost_submission_final.csv")
print(f"   2. Analyser l'importance des features dans feature_importance.csv")
print(f"   3. Soumettre sur Kaggle et vérifier le score")
if best_threshold != 0.5 and best_threshold_acc > oof_accuracy + 0.001:
    print(f"   4. Si besoin, tester la soumission avec seuil optimisé")
print(f"\n{'='*60}")
