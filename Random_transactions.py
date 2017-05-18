#--- import required libraries ---#
import warnings
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from decimal import Decimal
warnings.filterwarnings("ignore")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#--- open the file generate random observations add them to the file random_rows1 ---#
filename = 'creditcard.csv'
cc =  pd.read_csv("creditcard.csv")
cc.head()
cc = cc.rename(columns={'Class': 'Category'})
nlinesfile = 284808
nlinesrandomsample = 40000
lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False)
df2 = pd.read_csv(filename, skiprows=lines2skip)
df2.to_csv('random_rows1.csv')
cc =  pd.read_csv("random_rows1.csv")
cc= cc.rename(columns={'Class': 'Category'})
nor_obs = cc.loc[cc.Category==0]
ano_obs = cc.loc[cc.Category==1] 
print(len(nor_obs),len(ano_obs))
#--- open the file find the first 492 anomaly add them to the file random_rows1 ---#
cc =  pd.read_csv("creditcard.csv")
cc.head()
cc= cc.rename(columns={'Class': 'Category'})
nor_obs = cc.loc[cc.Category==0] 
ano_obs = cc.loc[cc.Category==1] 
print(len(nor_obs),len(ano_obs))
df1 = ano_obs.loc[492:, :]
df1=df1[:492]
type(df1),len(df1)
df1.to_csv('random_rows1.csv', mode='a', header=False)
cc =  pd.read_csv("random_rows1.csv")
cc= cc.rename(columns={'Class': 'Category'})
nor_obs = cc.loc[cc.Category==0]
ano_obs = cc.loc[cc.Category==1]
print(len(nor_obs),len(ano_obs))
