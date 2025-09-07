import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Toxic Comment Classification - Jigsaw Competition")
print("="*80)

train = pd.read_csv(r"C:\Users\W.I\Downloads\ML-Project\train.csv")
test = pd.read_csv(r"C:\Users\W.I\Downloads\ML-Project\test.csv")
test_labels = pd.read_csv(r"C:\Users\W.I\Downloads\ML-Project\test_labels.csv")

print("Dataset shapes:")
print(f"Training set: {train.shape}")
print(f"Test set: {test.shape}")
print(f"Test labels: {test_labels.shape}")

target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print("\nClass distribution in training data:")
for col in target_cols:
    ratio = train[col].mean() * 100
    print(f"{col}: {train[col].sum():,} positive samples ({ratio:.2f}%)")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"@\w+", "MENTION", text)
    text = re.sub(r"#\w+", "HASHTAG", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train['clean_text'] = train['comment_text'].apply(preprocess_text)
test['clean_text'] = test['comment_text'].apply(preprocess_text)
print("Text preprocessing completed")

blacklist = [
    "fuck", "shit", "ass", "bitch", "cunt", "dick", "piss", "cock", "pussy", 
    "bastard", "slut", "whore", "fag", "nigger", "retard", "idiot", "moron",
    "stupid", "damn", "hell", "suck", "kill", "die", "hate", "ugly", "fat"
]

def extract_features(text):
    features = {}
    text_str = str(text)
    
    features['text_length'] = len(text_str)
    features['word_count'] = len(text_str.split())
    features['avg_word_length'] = np.mean([len(word) for word in text_str.split()]) if text_str.split() else 0
    
    features['has_url'] = int("url" in text_str)
    features['has_mention'] = int("mention" in text_str)
    features['has_hashtag'] = int("hashtag" in text_str)
    features['caps_ratio'] = sum(1 for c in text_str if c.isupper()) / max(1, len(text_str))
    features['exclamation_count'] = text_str.count('!')
    features['question_count'] = text_str.count('?')
    
    features['blacklist_count'] = sum(1 for word in blacklist if word in text_str)
    features['offensive_ratio'] = features['blacklist_count'] / max(1, features['word_count'])
    
    return pd.Series(features)

print("Extracting features...")
train_features = train['clean_text'].apply(extract_features)
test_features = test['clean_text'].apply(extract_features)

train = pd.concat([train, train_features], axis=1)
test = pd.concat([test, test_features], axis=1)
print("Feature extraction completed")

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    stop_words='english'
)

X_train_text = vectorizer.fit_transform(train['clean_text'])
X_test_text = vectorizer.transform(test['clean_text'])
print("TF-IDF vectorization completed")

feature_cols = ['text_length', 'word_count', 'avg_word_length', 'has_url', 
                'has_mention', 'has_hashtag', 'caps_ratio', 'exclamation_count', 
                'question_count', 'blacklist_count', 'offensive_ratio']

extra_feats_train = csr_matrix(train[feature_cols].values.astype(float))
extra_feats_test = csr_matrix(test[feature_cols].values.astype(float))

X_train = hstack([X_train_text, extra_feats_train])
X_test = hstack([X_test_text, extra_feats_test])
print("Feature combination completed")

y_train = train['toxic']

mask = (test_labels['toxic'] != -1).values
X_test_eval = X_test[mask]
y_test_true = test_labels.loc[mask, 'toxic'].values

print(f"\nTest set class distribution:")
print(f"Class 0 (non-toxic): {(y_test_true == 0).sum():,}")
print(f"Class 1 (toxic): {(y_test_true == 1).sum():,}")

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample")
}

results = {}
print("\n" + "="*80)
print("Training and evaluating models")
print("="*80)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        roc_auc = roc_auc_score(y_test_true, y_pred_proba)
    else:
        y_pred = model.predict(X_test_eval)
        roc_auc = roc_auc_score(y_test_true, y_pred)
    
    acc = accuracy_score(y_test_true, y_pred)
    precision = precision_score(y_test_true, y_pred, zero_division=0)
    recall = recall_score(y_test_true, y_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_pred, zero_division=0)
    
    results[name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f" {name}")
    print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(f"   Precision: {precision:.4f} | Recall: {recall:.4f}")

print("\n" + "="*80)
print("Final Results")
print("="*80)

results_df = pd.DataFrame(results).T
print(results_df.round(4))

best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")

print("\n" + "="*80)
print("Visualizations")
print("="*80)

if hasattr(best_model, "predict_proba"):
    y_pred_proba = best_model.predict_proba(X_test_eval)[:, 1]
    y_pred_best = (y_pred_proba >= 0.5).astype(int)
else:
    y_pred_best = best_model.predict(X_test_eval)

cm = confusion_matrix(y_test_true, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Non-Toxic', 'Toxic'], 
            yticklabels=['Non-Toxic', 'Toxic'])
plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
models_names = list(results.keys())
f1_scores = [results[m]['f1'] for m in models_names]

plt.bar(models_names, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Model Comparison (F1 Score)', fontsize=14)
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "="*80)
print("Top 20 Important Features Analysis")
print("="*80)

tfidf_feature_names = vectorizer.get_feature_names_out()
extra_feature_names = feature_cols
all_feature_names = np.concatenate([tfidf_feature_names, extra_feature_names])

if best_model_name == "Random Forest":
    importance = best_model.feature_importances_
elif best_model_name == "Logistic Regression":
    importance = np.abs(best_model.coef_[0])
else:
    if hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_[0])
    else:
        print("Feature importance not available for this model type")
        importance = None

if importance is not None:
    top_indices = np.argsort(importance)[-20:]
    top_features = all_feature_names[top_indices]
    top_importance = importance[top_indices]

    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, 20))
    bars = plt.barh(range(20), top_importance, color=colors)
    
    plt.title('Top 20 Important Features', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.yticks(range(20), top_features, fontsize=10)
    plt.gca().invert_yaxis()
    
    for i, (value, bar) in enumerate(zip(top_importance, bars)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('top_20_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 20 Most Important Features:")
    print("-" * 50)
    for i, (feature, imp) in enumerate(zip(reversed(top_features), reversed(top_importance)), 1):
        print(f"{i:2d}. {feature}: {imp:.6f}")

print("\nSaving model and results...")

joblib.dump(best_model, 'best_toxic_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

results_df.to_csv('model_results.csv')
print("Model and results saved successfully")

print("\nCreating submission file...")

if hasattr(best_model, "predict_proba"):
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
else:
    test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    "id": test['id'],
    "toxic": test_preds
})

for col in target_cols[1:]:
    if col in train.columns:
        y_train_col = train[col]
        model_col = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model_col.fit(X_train, y_train_col)
        
        if hasattr(model_col, "predict_proba"):
            col_probs = model_col.predict_proba(X_test)[:, 1]
            col_preds = (col_probs >= 0.5).astype(int)
        else:
            col_preds = model_col.predict(X_test)
        
        submission[col] = col_preds

submission.to_csv("submission.csv", index=False)
print("Submission file saved: submission.csv")

print("\n" + "="*80)
print("Toxic comment classification completed successfully!")
print("="*80)