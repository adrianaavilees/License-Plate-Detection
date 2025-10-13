from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import cv2
import os
import joblib


# Path to your dataset folder
DATASET_DIR = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/Augmented_Letters"

data = []
labels = []

# Load each image and extract label from filename
for f in os.listdir(DATASET_DIR):
    if f.endswith(".png"):
        img_path = os.path.join(DATASET_DIR, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32)).flatten() / 255.0  # same preprocessing
        label = f.split("_")[0].upper()  # 'b_001.jpg' -> 'B'
        data.append(img)
        labels.append(label)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ✅ Train SVM
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# ✅ Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ✅ Save trained model
joblib.dump(clf, "svm_letters.pkl")
print("✅ Model saved as svm_letters.pkl")


clf = joblib.load("svm_letters.pkl")
