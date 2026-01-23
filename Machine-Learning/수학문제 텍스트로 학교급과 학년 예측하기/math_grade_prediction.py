import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import warnings

# Font setting for Mac
import matplotlib as mpl
mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 1. Load Data
try:
    train_df = pd.read_csv('data/school_train.csv')
    test_df = pd.read_csv('data/school_test.csv')
except FileNotFoundError:
    # Fallback to absolute path or current directory if needed
    import os
    base_path = '/Users/gimdabin/Machine-Learning/수학문제 텍스트로 학교급과 학년 예측하기'
    train_df = pd.read_csv(os.path.join(base_path, 'data/school_train.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'data/school_test.csv'))

print("Data Loaded Successfully")
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

# 2. Preprocessing
train_df['school_grade'] = train_df['school'] + ' ' + train_df['grade']
test_df['school_grade'] = test_df['school'] + ' ' + test_df['grade']

# Text cleaning
def remove_html_tags(text):
    if not isinstance(text, str):
        return str(text)
    return BeautifulSoup(text, 'html.parser').get_text()

print("Cleaning text...")
train_df['text_descriptions_clean'] = train_df['text_descriptions'].apply(remove_html_tags)
train_df['text_descriptions'] = train_df['text_descriptions_clean']
train_df.drop(columns=['text_descriptions_clean'], inplace=True)

test_df['text_descriptions_clean'] = test_df['text_descriptions'].apply(remove_html_tags)
test_df['text_descriptions'] = test_df['text_descriptions_clean']
test_df.drop(columns=['text_descriptions_clean'], inplace=True)

# Encode Target
label_encoder = LabelEncoder()
train_df['target'] = label_encoder.fit_transform(train_df['school_grade'])
test_df['target'] = label_encoder.transform(test_df['school_grade'])

# 3. Feature Extraction
print("Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 3),
    sublinear_tf=True,
    use_idf=True
)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text_descriptions'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['text_descriptions'])

print("Extracting Additional features...")
def extract_additional_features(df):
    features = pd.DataFrame()
    features['text_length'] = df['text_descriptions'].str.len()
    features['word_count'] = df['text_descriptions'].str.split().str.len()
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    
    # Mathematical and Numerical patterns
    features['num_count'] = df['text_descriptions'].str.count(r'\d+')
    features['has_fraction'] = df['text_descriptions'].str.contains(r'\d+/\d+').astype(int)
    features['has_decimal'] = df['text_descriptions'].str.contains(r'\d+\.\d+').astype(int)
    features['math_symbol_count'] = df['text_descriptions'].str.count(r'[+\-×÷=]')
    
    # Keywords
    features['has_geometry'] = df['text_descriptions'].str.contains(r'도형|삼각형|사각형|원|직사각형|정사각형').astype(int)
    features['has_measurement'] = df['text_descriptions'].str.contains(r'길이|넓이|부피|무게|시간').astype(int)
    features['has_calculation'] = df['text_descriptions'].str.contains(r'계산|구하|얼마').astype(int)
    features['has_comparison'] = df['text_descriptions'].str.contains(r'크다|작다|같다|많다|적다').astype(int)
    
    # Sentence structure
    features['sentence_count'] = df['text_descriptions'].str.count(r'[.!?]') + 1
    features['question_count'] = df['text_descriptions'].str.count(r'\?')
    features['parenthesis_count'] = df['text_descriptions'].str.count(r'[()\[\]]')
    
    return features

train_features = extract_additional_features(train_df)
test_features = extract_additional_features(test_df)

train_features = train_features.fillna(0)
test_features = test_features.fillna(0)

# Combine Features
X_train_combined = hstack([X_train_tfidf, train_features])
X_test_combined = hstack([X_test_tfidf, test_features])

print(f"Train Combined Shape: {X_train_combined.shape}")
print(f"Test Combined Shape: {X_test_combined.shape}")

# 4. Model Training and Evaluation
y_train = train_df['target']
y_test = test_df['target']

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": MultinomialNB()
}

results = {}

print("\nStarting Model Training...")
for name, model in models.items():
    print(f"\n{name} 학습 중입니다.")
    
    if name == "Naive Bayes":
        # Naive Bayes uses only TF-IDF features as per notebook logic
        model.fit(X_train_tfidf, y_train)
        train_pred = model.predict(X_train_tfidf)
        test_pred = model.predict(X_test_tfidf)
    else:
        model.fit(X_train_combined, y_train)
        train_pred = model.predict(X_train_combined)
        test_pred = model.predict(X_test_combined)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    
    print(f"학습 정확도 : {train_acc:.4f}")
    print(f"테스트 정확도 : {test_acc:.4f}")
    print(f"테스트 F1-Score : {test_f1:.4f}")

# Select Best Model
best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_model = results[best_model_name]['model']

print(f"\n최고 성능 모델 : {best_model_name}")
print(f"검증 정확도 : {results[best_model_name]['test_acc']:.4f}")
print(f"검증 f1-Score : {results[best_model_name]['test_f1']:.4f}")

# Predictions using Best Model
if best_model_name == "Naive Bayes":
    y_test_pred = best_model.predict(X_test_tfidf)
else:
    y_test_pred = best_model.predict(X_test_combined)

print("\n===Classification Report===")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# 5. Ensemble Model
print("\n=== Ensemble (Voting Classifier) ===")
# Re-instantiate models for ensemble
lr_for_ensemble = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
rf_for_ensemble = RandomForestClassifier(n_estimators=100, random_state=42)
nb_for_ensemble = MultinomialNB()

ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr_for_ensemble),
        ('rf', rf_for_ensemble),
        ('nb', nb_for_ensemble)
    ],
    voting='soft'
)

# Note: The notebook fits ensemble on X_train_combined.
# MultinomialNB will work if X_train_combined is non-negative.
# TF-IDF is non-negative. Additional features (lengths, counts) are non-negative.
# So this is valid.
ensemble_model.fit(X_train_combined, y_train)
ensemble_predictions = ensemble_model.predict(X_test_combined)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
ensemble_f1_score = f1_score(y_test, ensemble_predictions, average='weighted')

print(f"앙상블 모델 정확도 : {ensemble_accuracy:.4f}")
print(f"앙상블 모델 F1 Score : {ensemble_f1_score:.4f}")
