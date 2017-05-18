#--- import required libraries ---#
import time
#--- start counting time ---#
start_time = time.time()
import csv
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import roc_curve, auc
import pandas as pd
from PIL import Image
import numpy as np
from sklearn import svm
warnings.filterwarnings('ignore')
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#--- reading csv file and divide the data into normal and outliers ---#
cc =  pd.read_csv('random_rows2.csv')
cc.head() 
cc= cc.rename(columns={'Class': 'Category'})
normal_observations = cc.loc[cc.Category==0]
abnormal_observations= cc.loc[cc.Category==1]
normal_data = normal_observations.iloc[:,:].values
abnormal_data= abnormal_observations.iloc[:,:].values
np.shape(normal_data), np.shape(abnormal_data)
#--- divide normal data into training and testing, features and classes ---#
X_train = normal_data[:30000,:-1]
Y_train = normal_data[:30000,-1]
xnormal_test = normal_data[30000:,:-1]
ynormal_test = normal_data[30000:,-1]
#--- append the outliers to testing data ---#
X_test=np.concatenate((xnormal_test, abnormal_data[:,:-1]), axis=0)
Y_test=np.concatenate((ynormal_test, abnormal_data[:,-1]), axis=0)
Ys_test=[[1-int(i), int(i)] for i in Y_test]
np.shape(Ys_test)
print(Y_test.shape,type(Y_test),type(Ys_test))
#--- creating lists to store the values of measurements ---#
measurment_list=[]
recall_list=[]
fallout_list=[]
fpr_list=[]
tpr_list=[]
roc_auc_list=[]
#--- define range of nu and gamma parameters ---#
nu =  np.linspace(0.01,0.09,10)
gamma = np.linspace(0.001,0.0001,10)
#--- create predictions list to store the classifiers output ---#
predictions=[]
#--- train and test the classifiers with different parameters value and update the lists ---#
for j in range(len(nu)):
    oneclass = svm.OneClassSVM(kernel='rbf', gamma=gamma[j], nu=nu[j])
    oneclass.fit(X_train)
    predictions.append(oneclass.predict(X_test))    
    Y_score = oneclass.decision_function(X_test)
    Ys_test=np.array(Ys_test)
    fpr, tpr, _ = roc_curve(Ys_test[:,0], Y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)
#--- check the confusion matrix componants ---#        
for predict in predictions:
    TP = FN = FP = TN = 0
    for i in range(len(Y_test)):
        if Ys_test[i][0] == 1 and predict[i] == 1:
            TP = TP+1
        elif Ys_test[i][0] == 1 and predict[i] == -1:
            FN = FN+1
        elif Ys_test[i][1] == 1 and predict[i] == 1:
            FP = FP+1
        else:
            TN = TN +1
    print("results:", TP, FN, FP, TN)
    measurment_list.append([TP,FN,FP,TN])
    recall_list.append(TP/(TP+FN))
    fallout_list.append(FP/(FP+TN))
#--- plot ROC curves for all classifiers output ---#
colors = ['red','green','blue','purple','navy','cyan','magenta','yellow','orange','grey']
for k in range(len(fpr_list)):
    plt.plot(fpr_list[k], tpr_list[k], color=colors[k],
             lw=2, label='%0.2f, %0.4f, %0.2f' % (nu[k],gamma[k],roc_auc_list[k]))
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.axis([0,1,0,1])
#--- stop counting time ---#
end_time = time.time()
#--- displaying the values of lists ---#
print(measurment_list)
print('_________________________________________')
print(recall_list)
print('_________________________________________')
print(fallout_list)
print('_________________________________________')
print(fpr_list)
print('_________________________________________')
print(tpr_list)
print('_________________________________________')
print(roc_auc_list)
#--- show the plot and print the time ---#
plt.show()    
print('Elapsed time was %g seconds' % (end_time - start_time))

