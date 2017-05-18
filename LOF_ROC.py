#--- import required libraries ---#
import time
#--- start counting time ---#
start_time = time.time()
from PIL import Image
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import cm
import pandas as pd
from sklearn.metrics import roc_curve, auc
#--- creating figure and lists of measurements ---#
fig=plt.figure(1)
measurment_list=[]
recall_list=[]
fallout_list=[]
fpr_list=[]
tpr_list=[]
roc_auc_list=[]  
#--- reading csv file and find all the normal observations to train the model ---#
data = pd.read_csv('random_rows2.csv')
X = data.iloc[:,:-1].values
y = data['Class'].values
#--- define range of k value parameters ---#
knn=np.linspace(10,100,10)
#--- train the classifier with normal observations ---#
for k in range(len(knn)):
    clf = LocalOutlierFactor(n_neighbors=int(knn[k]))
    predictions = clf.fit_predict(X)
#--- check the confusion matrix componants ---#    
    TP = FN = FP = TN = 0
    for i in range(len(predictions)):
        if y[i] == 0 and predictions[i] == 1:
            TP = TP+1
        elif y[i] == 0 and predictions[i] == -1:
            FN = FN+1
        elif y[i] == 1 and predictions[i] == 1:
            FP = FP+1
        else:
            TN = TN +1
#--- calculat the values of measurements and save the results in lists ---#            
    measurment_list.append([TP,FN,FP,TN])
    recall = TP/(TP+FN)
    recall_list.append(recall)
    fallout = FP/(FP+TN)
    fallout_list.append(fallout)
#--- test the model by finding the score, find the ROC curve and save the values in the lists ---#
    Y_score = clf._decision_function(X)      
    fpr, tpr, _ = roc_curve(y, Y_score)
    roc_auc = auc(fpr, tpr)  
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)
#--- plot ROC curves for all classifiers output ---#
colors = ["red", "green", "blue", "purple","navy","cyan","magenta","yellow","orange","grey"]
for k in range(len(fpr_list)):
    plt.plot(fpr_list[k], tpr_list[k], color=colors[k],
             lw=2, label='%0.3s, %0.2f' % (knn[k],roc_auc_list[k]))
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="upper left")
#--- show the plot and print the time ---#    
end_time = time.time()
#--- save the plot  ---#
plt.savefig('LOF_ROC.png')
img = Image.open('LOF_ROC.png')
img.show()
print("Elapsed time was %g seconds" % (end_time - start_time))
