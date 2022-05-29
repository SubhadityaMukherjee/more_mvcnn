import pickle
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

with open("cons.pkl", "rb") as f:
    out, y_prob = pickle.load(f)

cm = confusion_matrix(y_prob, out, normalize='true')
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("results/confusion_matrix.png")
print("saved")
