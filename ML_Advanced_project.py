import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore') 
import time
import psutil
import gc

print("=" * 100)
print(" Dataset Info")
print("=" * 100)

print("Loading data...")
train = pd.read_csv(r"C:\Users\W.I\Downloads\ML-Project\Advanced\all_data.csv")
test = pd.read_csv(r"C:\Users\W.I\Downloads\ML-Project\Advanced\test_public_expanded.csv")
sample_sub = pd.read_csv(r"C:\Users\W.I\Downloads\ML-Project\Advanced\sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("\nSample data:\n", train[['id', 'comment_text', 'split', 'created_date']].head(3))
print("=" * 100)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"@\w+", "MENTION", text)
    text = re.sub(r"#\w+", "HASHTAG", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

print("Preprocessing text...")
train['clean_text'] = train['comment_text'].apply(preprocess_text)
test['clean_text'] = test['comment_text'].apply(preprocess_text)
print(" Preprocessing done. Added column [clean_text].")

del train['comment_text'], test['comment_text']
gc.collect()

blacklist = ["fuck", "shit", "asshole", "bitch", "cunt", "dick", "piss", "cock", "pussy", "bastard", 
             "slut", "whore", "douche", "fag", "faggot", "nigger", "nigga", "retard", "idiot", "moron"]

def rule_based_features(text):
    features = {}
    text_str = str(text)
    features['has_url'] = int("url" in text_str)
    features['has_mention'] = int("mention" in text_str)
    features['has_hashtag'] = int("hashtag" in text_str)
    features['blacklist_count'] = sum(1 for word in blacklist if word in text_str)
    features['text_length'] = len(text_str)
    features['caps_ratio'] = sum(1 for c in text_str if c.isupper()) / (len(text_str) + 1e-5)
    features['exclamation_count'] = text_str.count('!')
    features['question_count'] = text_str.count('?')
    return pd.Series(features)

print("Extracting rule-based features...")
sample_size = min(100000, len(train))
train_sample = train.sample(n=sample_size, random_state=42)
train_features = train_sample['clean_text'].apply(rule_based_features)
train_sample = pd.concat([train_sample, train_features], axis=1)

test_sample_size = min(50000, len(test))
test_sample = test.sample(n=test_sample_size, random_state=42)
test_features = test_sample['clean_text'].apply(rule_based_features)
test_sample = pd.concat([test_sample, test_features], axis=1)

print(" Rule-based features added to samples.")

del train_features, test_features
gc.collect()

print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=5, max_df=0.9)
X_train_text = vectorizer.fit_transform(train_sample['clean_text'])
X_test_text = vectorizer.transform(test_sample['clean_text'])

extra_feats_train = csr_matrix(train_sample[['has_url','has_mention','has_hashtag','blacklist_count',
                                      'text_length','caps_ratio','exclamation_count', 'question_count']].values.astype('float32'))
extra_feats_test = csr_matrix(test_sample[['has_url','has_mention','has_hashtag','blacklist_count',
                                    'text_length','caps_ratio','exclamation_count', 'question_count']].values.astype('float32'))

X_train_base = hstack([X_train_text, extra_feats_train])
X_test_final = hstack([X_test_text, extra_feats_test])

del X_train_text, X_test_text, extra_feats_train, extra_feats_test
gc.collect()

rare_targets = ['severe_toxicity', 'threat', 'sexual_explicit', 'obscene']
for col in rare_targets:
    if col in train_sample.columns:
        train_sample[col] = train_sample[col].fillna(0).astype('float32')

train_sample['is_toxic_rare'] = train_sample[rare_targets].any(axis=1).astype('int8')

main_targets = ['toxicity', 'insult', 'identity_attack']

submission = sample_sub.copy()
results = {}

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}% used, {memory.available / (1024**3):.1f}GB available")

def train_model(target, X_train, y_train):
    print(f"\n{'='*60}")
    print(f"TRAINING FOR TARGET: {target.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    check_memory()
    
    y_train = y_train.astype('int8')
    
    class_counts = y_train.value_counts()
    print(f"Class distribution:")
    print(f"Class 0: {class_counts.get(0, 0):,}")
    print(f"Class 1: {class_counts.get(1, 0):,}")
    if class_counts.get(1, 0) > 0:
        print(f"Ratio: 1:{class_counts.get(0, 0)/class_counts.get(1, 0):.1f}")
    
    if class_counts.get(1, 0) < 1000:
        print(f"⚠  SKIPPING {target} - Only {class_counts.get(1, 0)} positive examples")
        return None
    
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    if class_counts.get(1, 0) / class_counts.get(0, 0) < 0.3:
        print(" Applying RandomUnderSampler for class balancing...")
        rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
        X_train_res, y_train_res = rus.fit_resample(X_train_sub, y_train_sub)
        print(f"After balancing: {X_train_res.shape[0]:,} samples")
    else:
        print(" Using original data (balanced enough)")
        X_train_res, y_train_res = X_train_sub, y_train_sub

    print("🏋 Training Logistic Regression model...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        random_state=42,
        solver='liblinear',
        C=0.1,
        n_jobs=1
    )
    
    model.fit(X_train_res, y_train_res)

    print(" Evaluating model...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    result = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': best_threshold,
        'model': model,
        'y_true': y_val,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\n RESULTS FOR {target.upper()}:")
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    training_time = time.time() - start_time
    print(f" Training time: {training_time:.2f} seconds")
    
    return result

print("\n" + "="*100)
print(" STARTING TRAINING (USING SAMPLES)")
print("="*100)

for target in main_targets:
    try:
        if target not in train_sample.columns:
            print(f"⚠  Target {target} not found in data, skipping...")
            continue
            
        print(f"\n{'#'*80}")
        print(f"PROCESSING TARGET: {target}")
        print(f"{'#'*80}")
        
        y_train_target = (train_sample[target].fillna(0) >= 0.5).astype(int)
        result = train_model(target, X_train_base, y_train_target)
        
        if result is not None:
            results[target] = result
            print(f" Training completed for {target}")
            
    except Exception as e:
        print(f" ERROR training {target}: {str(e)}")

print("\n" + "="*100)
print(" SAVING RESULTS")
print("="*100)

for target in submission.columns:
    if target != 'id':
        submission[target] = 0

submission.to_csv("submission_final.csv", index=False)
print(" Submission file saved as submission_final.csv")

if results:
    print(" Creating visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    targets = list(results.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_data = {
        'Accuracy': [results[target]['accuracy'] for target in targets],
        'Precision': [results[target]['precision'] for target in targets],
        'Recall': [results[target]['recall'] for target in targets],
        'F1-Score': [results[target]['f1'] for target in targets]
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (metric_name, color) in enumerate(zip(metrics_names, colors), 1):
        plt.subplot(2, 2, i)
        plt.bar(targets, metrics_data[metric_name], color=color, alpha=0.7)
        plt.title(f'{metric_name} by Target', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        for j, value in enumerate(metrics_data[metric_name]):
            plt.text(j, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for target in targets:
        fpr, tpr, _ = roc_curve(results[target]['y_true'], results[target]['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{target} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 5))
    
    if n_targets == 1:
        axes = [axes]
    
    for i, target in enumerate(targets):
        cm = confusion_matrix(results[target]['y_true'], results[target]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                   cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
        axes[i].set_title(f'Confusion Matrix - {target}', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    for target in targets:
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (results[target]['y_pred_proba'] >= threshold).astype(int)
            f1_scores.append(f1_score(results[target]['y_true'], y_pred, zero_division=0))
        
        plt.plot(thresholds, f1_scores, marker='o', label=target, linewidth=2)
        best_idx = np.argmax(f1_scores)
        plt.scatter(thresholds[best_idx], f1_scores[best_idx], color='red', s=100, zorder=5)
        plt.text(thresholds[best_idx], f1_scores[best_idx] + 0.02, 
                f'Best: {thresholds[best_idx]:.2f}', ha='center', fontweight='bold')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('Threshold Optimization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    class_distribution = []
    
    for target in targets:
        y_train_target = (train_sample[target].fillna(0) >= 0.5).astype(int)
        counts = y_train_target.value_counts()
        class_distribution.append([counts.get(0, 0), counts.get(1, 0)])
    
    class_distribution = np.array(class_distribution)
    
    plt.bar(np.arange(len(targets)) - 0.2, class_distribution[:, 0], width=0.4, 
            label='Class 0 (Negative)', alpha=0.7, color='skyblue')
    plt.bar(np.arange(len(targets)) + 0.2, class_distribution[:, 1], width=0.4, 
            label='Class 1 (Positive)', alpha=0.7, color='lightcoral')
    
    plt.xlabel('Target Variables', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(len(targets)), targets, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved:")
    print("   - model_performance.png")
    print("   - roc_curves.png")
    print("   - confusion_matrices.png")
    print("   - threshold_optimization.png")
    print("   - class_distribution.png")

else:
    print("⚠  No results to visualize")

print("\n" + "="*100)
print(" FINAL RESULTS SUMMARY:")
print("-" * 80)
for target in results.keys():
    metrics = results[target]
    print(f"{target:20} | Acc: {metrics['accuracy']:.3f} | Prec: {metrics['precision']:.3f} | "
          f"Rec: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | Thr: {metrics['threshold']:.2f}")

print("="*100)
print(" PROCESS COMPLETED SUCCESSFULLY!")
print("="*100)