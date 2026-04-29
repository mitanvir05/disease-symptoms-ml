import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

print("Loading data for evaluation...")
df = pd.read_csv('mtsamples.csv')

# 1. Clean Data
df = df[['transcription', 'medical_specialty']].dropna()

# 2. Map Labels
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

# 3. Balance Data

dfs = []
for specialty, group in df.groupby('medical_specialty'):
   
    dfs.append(group.sample(min(len(group), 100))) 

df = pd.concat(dfs).reset_index(drop=True)

X = df['transcription']
y = df['medical_specialty']

# 4. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training High-Performance SVC on {len(X_train)} samples...")

# 5. Define The "High Accuracy" Pipeline
# - TF-IDF with (1,2) grams learns phrases like "chest pain"

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
classifier = CalibratedClassifierCV(LinearSVC(dual="auto", class_weight='balanced'))

model = make_pipeline(vectorizer, classifier)
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Generate Report
acc = accuracy_score(y_test, y_pred) * 100
print("\n--- HIGH-PERFORMANCE MODEL REPORT ---")
print(f"Overall Accuracy: {acc:.2f}%\n")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Generate Confusion Matrix Chart
try:
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (LinearSVC - {acc:.1f}% Accuracy)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('evaluation_chart_svc.png')
    print("✅ Proof Chart saved as 'evaluation_chart_svc.png'")
except Exception as e:
    print(f"Chart error: {e}")