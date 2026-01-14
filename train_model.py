import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

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

# --- BULLETPROOF BALANCING (The Fix) ---
# Instead of using .apply() which causes the KeyError, we loop manually.
dfs = []
for specialty, group in df.groupby('medical_specialty'):
    # Sample up to 100 rows per specialty
    dfs.append(group.sample(min(len(group), 100))) 

df_balanced = pd.concat(dfs).reset_index(drop=True)

print(f"Training on {len(df_balanced)} balanced records...")

X = df_balanced['transcription']
y = df_balanced['medical_specialty']

# --- Pipeline ---
# Using LinearSVC for higher accuracy
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
classifier = CalibratedClassifierCV(LinearSVC(dual="auto", class_weight='balanced'))

model = make_pipeline(vectorizer, classifier)

model.fit(X, y)

joblib.dump(model, 'symptom_model.pkl')
print("✅ High-Accuracy SVC Model Trained!")