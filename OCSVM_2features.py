#--- plot all the features in 2d by taking 2 features at one time ---#
#--- import required libraries ---#
get_ipython().magic('matplotlib inline')
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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#--- read csv file ---#
with open('random_rows.csv', 'r') as f:
    reader = csv.reader(f)
    csv_values = list(reader)
#--- convert data type from string to float ---#
def read_lines():
    with open('random_rows.csv', 'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]
#--- divide observations into training and testing data ---#
def observations(l):
    a = 2*(len(l)/3) + (len(l)%3)
    return(l[:a],l[a:])
#--- values for meshgrid  ---#
xx, yy= np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
#--- Classify observations into normal and outliers ---#
X = []; X_train = []; X_test = []; X_outliers = []
for i in range(len(csv_values)):
    if csv_values[i][-1] == '0':
        X.append(csv_values[i][:-1]) 
    else:
        X_outliers.append(csv_values[i][:-1])
        
print(len(X),len(X_outliers),type(X),type(X_outliers))
#--- divide observations into training and testing ---#
X_train, X_test = observations(X)
#--- convert lists to arrays ---#
X=np.array(X)
X_train1=np.array(X_train)
X_test1 = np.array(X_test)
X_outliers1= np.array(X_outliers)
fig=plt.figure(1)
#--- loop through the feautres ---#
for i in range(27):
    X_train=X_train1[:,i:i+2]
    X_test = X_test1[:,i:i+2]
    X_outliers= X_outliers1[:,i:i+2]
#--- fit the model ---#
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
#--- predict the output of the classifier ---#
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#--- plot the line, the points, and the nearest vectors to the plane ---#
    Z = Z.reshape(xx.shape)
    fig.add_subplot(9,3,i+1)
    fig.set_figheight(20)
    fig.set_figwidth(16)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
#--- draw the scatter with all different features ---#
    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, marker=">")
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, marker=",")
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
fig.savefig('27features.png')
#--- Histogram of the Observations ---#
plt.figure(figsize=(10,5))
barlist=plt.bar([1,2,3], [len(X_train),len(X_test),len(X_outliers)],width=0.5, alpha=0.7, align='center')
plt.subplots_adjust(left=0.1, right=0.5, top=0.9, bottom=0.1)
barlist[0].set_color('r')
barlist[1].set_color('y')
plt.title('Training, Testing and Outliers Observations Clusters')
plt.ylabel('Number of Observations')
plt.xlabel('Observation Cluster')
plt.xticks([1,2,3],['Training','Testing','Outliers'],rotation=45)
plt.grid(True,color='k')
plt.show()
#--- pick two feaures out of 28 ---#
X_train=X_train1[:,0:2]
X_test = X_test1[:,0:2]
X_outliers= X_outliers1[:,0:2]
#--- fit the model ---#
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# plot the line, the points, and the nearest vectors to the plane
Z = Z.reshape(xx.shape)
plt.subplots_adjust(wspace=0, hspace=0)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
s = 40
#--- features plotting ---#
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, marker=">")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, marker=",")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["Training observations",
            "Testing observations", "Outliers"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

