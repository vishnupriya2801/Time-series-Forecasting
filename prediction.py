import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb

import os
print(os.listdir("../input"))

data  = pd.read_csv('../input/sales.csv')

data.describe()
#checking for missing values
data.isnull().values.any()

#normalizing the data
data_norm = data.copy()
data_norm[['Normalized {}'.format(i) for i in range(0,52)]].head()
data_norm = data_norm[['Normalized {}'.format(i) for i in range(0,52)]]
data_norm.head()

def dddraw(X_reduced,name):
    from mpl_toolkits.mplot3d import Axes3D
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired)
    titel="First three directions of "+name 
    ax.set_title(titel)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()
    
    
 #prediction for products;
from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.cluster import KMeans,Birch
import statsmodels.formula.api as sm
from scipy import linalg
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return ( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ).round()

n_col=50
X = sales.drop(['Product_Code','W51','Normalized 51'],axis=1)
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)

    
Y=sales['W51']
X=X.fillna(value=0)  # NaN

names = [
         'PCA',
         'FastICA',
         'Gauss',
         'KMeans',
         'SparsePCA',
         'SparseRP',
         'Birch',
         'NMF'   
          
        ]

classifiers = [
    
    PCA(n_components=n_col),
    FastICA(n_components=n_col),
    GaussianRandomProjection(n_components=3),
    KMeans(n_clusters=n_col),
    SparseRandomProjection(n_components=n_col, dense_output=True),
    Birch(branching_factor=10, n_clusters=7, threshold=0.5),
    NMF(n_components=n_col)
    
]
correction= [1,1,0,0,0,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    Xr=clf.fit_transform(X,Y)
    dddraw(Xr,name)
    res = sm.OLS(Y,Xr).fit()
    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction
    
    print('Ypredict',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction
    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y)) #
