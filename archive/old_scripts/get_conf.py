#%%
import pickle
from tkinter import Y
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
#%%
with open("cons.pkl", "rb") as f:
    out, y_prob = pickle.load(f)
#%%
len(set([np.argmax(x) for x in y_prob[0]]))
# [np.argmax(x) for x in y_prob]
#%%
print(out,y_prob)
out = np.array(out[0])
# print(out[0].shape,y_prob[0][:,0])

cm = confusion_matrix(y_prob, out, normalize='true')
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("results/confusion_matrix.png")
print("saved")
