#--- import required libraries ---#
import csv
import scipy
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#--- read csv file ---#
with open('random_rows3.csv', 'r') as f:
    reader = csv.reader(f)
    csv_values = list(reader)
#--- convert data type from string to float ---#
def read_lines():
    with open('random_rows3.csv', 'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]
#--- values for meshgrid  ---#
xx, yy= np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
#--- Classify observations into normal and outliers ---#
X = []; X_train = []; X_test = []; X_outliers = []
for i in range(len(csv_values)):
    if csv_values[i][-1] == '0':
        X.append(csv_values[i][:-1]) 
    else:
        X_outliers.append(csv_values[i][:-1])
#--- convert lists to arrays ---#
X=np.array(X)
a=X[:,[0,2]]
X=a
fig=plt.figure(1)
X_outliers1= np.array(X_outliers)
b=X_outliers1[:,[0,1]]
X_outliers1=b
X = np.r_[X, X_outliers1]
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
y_pred_outliers = y_pred[39925:]
#--- plot the level sets of the decision function ---#
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplots_adjust(wspace=0, hspace=0)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=5, colors='darkred')
#--- plot the values ---#
a = plt.scatter(X[:39925, 0], X[:39925, 1], c='gold', s=40, marker="s", edgecolor='black')
b = plt.scatter(X[39925:, 0], X[39925:, 1], c='blueviolet', s=40, edgecolor='black')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
legend = plt.legend([a, b],
           ["Normal observations",
            "Outliers"],
              loc="upper left",shadow=False, fontsize='10',frameon=True)
legend.get_frame().set_alpha(1)
legend.get_frame().set_edgecolor('k')          
           
#--- save the plot and display it ---#
plt.savefig('LOF_2f.png')
img = Image.open('LOF_2f.png')
img.show()
