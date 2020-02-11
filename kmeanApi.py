
import pandas as pd
from sklearn import *

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.cluster import KMeans



class  msApi:
    def trainModel(self,split_Size):
        df = pd.read_csv ('./data/WineDataSet.csv' , header=None ,
                          names=["Class" , "Alcohol" , "Malic_Acid" , "Ash" , "Ash_Alcalinity" , "Magnesium" ,
                                 "Total_Phenols" , "Flavanoids" , "Nonflavanoid_Phenols" , "Proanthocyanins" ,
                                 "Colour_Intensity" , "Hue" , "OD280/OD315_of_diluted_wines" , "Proline"])

        dataframe = df.iloc[1:]
        for x in dataframe.columns:
            dataframe[x] = dataframe[x].astype (float)
        cols = dataframe.columns[0:]
        train , test = train_test_split (dataframe , test_size=0.2)
        cluster = KMeans (n_clusters=5)
        train["cluster"] = cluster.fit_predict (train[train.columns[0:]])
        joblib.dump (cluster , 'cluster_model.pkl')
        return "Model Created"

    def predict(self,criteria):
        nb_model = joblib.load ('cluster_model.pkl')
        X = [[float (criteria['Class']) , float (criteria['Alcohol']) , float (criteria['Malic_Acid']) ,
              float (criteria['Ash']) ,
              float (criteria['Ash_Alcalinity']) , float (criteria['Magnesium']) , float (criteria['Total_Phenols']) ,
              float (criteria['Flavanoids']) ,
              float (criteria['Nonflavanoid_Phenols']) , float (criteria['Proanthocyanins']) ,
              float (criteria['Colour_Intensity']) , float (criteria['Hue']),
              float (criteria['OD280/OD315_of_diluted_wines']), float (criteria['Proline'])
              ]]
        print(X)
        cluster=nb_model.predict (X)

        return  cluster