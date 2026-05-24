import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


cat_path = "PetImages/Cat"
dog_path = "PetImages/Dog"

data = []
labels = []

IMG_SIZE = 64


for img_name in os.listdir(cat_path)[:300]:

    try:
        img_path = os.path.join(cat_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.flatten()

        data.append(img)

        labels.append(0)

    except:
        pass


for img_name in os.listdir(dog_path)[:300]:

    try:
        img_path = os.path.join(dog_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.flatten()

        data.append(img)

        labels.append(1)

    except:
        pass


X = np.array(data)
y = np.array(labels)

print("Total Images Loaded:", len(X))


X = X / 255.0


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
