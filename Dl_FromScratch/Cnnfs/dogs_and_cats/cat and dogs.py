import os
import cv2
import numpy as np
import pickle

DATADIR = os.path.join(os.path.dirname(__file__), "test_set")
CATEGORIES = ["cats", "dogs"]

IMG_SIZE = 100   

X = []
y = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    label = CATEGORIES.index(category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            X.append(img_array)
            y.append(label)

        except Exception as e:
            pass

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)



pickle.dump(X, open(os.path.join(os.path.dirname(__file__), "test_datas.pickle"), "wb"))
pickle.dump(y, open(os.path.join(os.path.dirname(__file__), "test_labels.pickle"), "wb"))

print(X.shape)
print(y.shape)
print(y[:10])

