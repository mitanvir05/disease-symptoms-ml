import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import scipy.sparse

# 1. Advanced Text Cleaning Function

def clean_medical_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading and cleaning dataset...")
df = pd.read_csv('mtsamples.csv')
df = df[['transcription', 'medical_specialty']].dropna()

# --- Mapping ---
label_mapping = {
    ' Surgery': 'General Surgery',
    ' Consult - History and Phy.': 'General Medicine',
    ' Orthopedic': 'Orthopedics',
    ' Neurology': 'Neurology',
    ' Gastroenterology': 'Gastroenterology',
    ' Urology': 'Urology',
    ' ENT - Otolaryngology': 'ENT',
    ' Obstetrics / Gynecology': 'Gynecology',
    ' Hematology - Oncology': 'Oncology',
    ' Ophthalmology': 'Ophthalmology',
    ' Nephrology': 'Nephrology',
    ' Pediatrics - Neonatal': 'Pediatrics',
    ' Pain Management': 'General Medicine',
    ' Emergency Room Reports': 'General Medicine'
}

df['medical_specialty'] = df['medical_specialty'].map(label_mapping)
df = df.dropna(subset=['medical_specialty'])

# Apply cleaning
print("Preprocessing medical text...")
df['transcription'] = df['transcription'].apply(clean_medical_text)

# --- Optimized Data Sampling ---
# Increased limit to 150 rows per specialty to provide more training context.
dfs = []
for specialty, group in df.groupby('medical_specialty'):
    dfs.append(group.sample(min(len(group), 150), random_state=42)) 

df_balanced = pd.concat(dfs).reset_index(drop=True)
print(f"Training on {len(df_balanced)} optimized records...")

X = df_balanced['transcription']
y = df_balanced['medical_specialty']

# --- Hyperparameter Tuning (GridSearch) ---
# We define a pipeline and let GridSearchCV find the best 'C' and 'ngram' settings.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=8000)),
    ('clf', LinearSVC(dual="auto", class_weight='balanced', max_iter=2000))
])

param_grid = {
    'tfidf__ngram_range': [(1, 2)], # Testing single words and word pairs
    'tfidf__min_df': [2, 3],        # Ignore very rare words
    'clf__C': [0.1, 1, 10]          # Testing penalty strengths
}

print("Running Grid Search to find best accuracy settings...")
# cv=3 means it will validate the model 3 times for every setting to ensure stability
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X, y)

print(f"Best Accuracy found in Search: {grid.best_score_:.4f}")

# --- Final Calibration ---

best_params = grid.best_params_
print(f"Best Parameters: {best_params}")

# Re-build best vectorizer
best_tfidf = TfidfVectorizer(
    stop_words='english', 
    max_features=8000,
    ngram_range=best_params['tfidf__ngram_range'],
    min_df=best_params['tfidf__min_df']
)

# Re-build and calibrate the best classifier
best_clf = LinearSVC(
    dual="auto", 
    class_weight='balanced', 
    C=best_params['clf__C'],
    max_iter=3000
)
calibrated_clf = CalibratedClassifierCV(best_clf, cv=5)

# Final fit
X_vec = best_tfidf.fit_transform(X)
calibrated_clf.fit(X_vec, y)

# Combine into a final pipeline for saving
final_model = Pipeline([
    ('tfidf', best_tfidf),
    ('clf', calibrated_clf)
])

joblib.dump(final_model, 'symptom_model.pkl')
print("✅ Optimized High-Accuracy Model Saved!")