from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import os
import joblib

data = []
labels = []

DATASET_DIR = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/Crops_numbers"

for f in os.listdir(DATASET_DIR):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join(DATASET_DIR, f), cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(os.path.join("Crops_numbers", f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32)).flatten() / 255.0
        data.append(img)
        labels.append(int(f.split("_")[0]))

X = np.array(data)
y = np.array(labels)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

#clf = svm.SVC(kernel='linear')
clf = svm.SVC(kernel='rbf', C=10, gamma=0.01)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


joblib.dump(clf, "svm_digits.pkl")




