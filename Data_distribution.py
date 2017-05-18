#--- import required liberaries ---#
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#--- read the file---# 
df=pd.read_csv('random_rows1.csv')
#--- fetch normal observations ---#
normal=df['Amount']
#--- fetch outliers ---#
outlier=df[df['Class']==1]['Amount']
#--- randomize the data, introduce new zero values to ensure uniform distribution ---#
np.random.shuffle(outlier.index.values)
zerolist=np.zeros(len(normal))
zerolist[0:len(outlier)]=outlier.values
np.random.shuffle(zerolist)
ind=[]
shuffledOutlier=[]
#--- select non zero values to plot ---#
for i in range(len(zerolist)):
    if zerolist[i]!=0:
        ind.append(i)
        shuffledOutlier.append(zerolist[i])
#--- plotting the results ---#
plt.figure(figsize=(20,9))
plt.ylim(0,4000)
plt.plot(normal,label='Normal observation',color='gray')
plt.plot(ind,shuffledOutlier,'ko',label='Outlier')
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('axes',edgecolor='r')
plt.ylabel('Amount of Transactions', fontsize=22)
plt.xlabel('Observations', fontsize=22)
leg = plt.legend(loc='upper left',prop={'size':22},frameon=True)
leg.get_frame().set_edgecolor('k')
plt.margins(0,0)
plt.savefig('distribution.png')
plt.show()
