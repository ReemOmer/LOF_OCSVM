# LOF_OCSVM
This folder contains python codes that used to compare between Local Outlier Factor algorithm and One Class Support Vector Machine on detecting outliers in a credit card transactions dataset.

The process starts by reading the credit card transactions file and then select 40492 random transactions from the dataset. The whole set contains 284,807 transactions each one has 28 features. All the data is numarical value.

To run the file you should start by: 

Random_transactions.py : read creditcard.csv and select the transaction randomly also it will select the features,

LOF-2features.py : will display all the combination of features in 2dimensional space for Local Outlier Factor (LOF) model,

OCSVM_2features.py: will display all the combination of features in 2dimensional space for Once Class Support Vector Machine (OCSVM), Histogram of the data division is also there,

Data_distribution.py : this file contains the code of displaying amounts of transactions,

LOF_ROC.py : this file is to find the values for performance measurements of LOF, plot the ROC curve and calculate the time for all the parameter values (k),

OCSVM_ROC.py : this file is to find the values for performance measurements of OCSVM, plot the ROC curve and calculate the time for all the parameters values (gamma and nu).
