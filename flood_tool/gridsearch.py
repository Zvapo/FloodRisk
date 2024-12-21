"""Analysis tools."""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif


from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from .transformer import Tr


__all__ = ['MyGridSearch']

class MyGridSearch(object):
    def __init__(self, postcode_file='', labelled_samples='', sample_labels=['riskLabel','medianPrice'],
                 household_file='', rain_file='', river_file=''):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        if postcode_file == '':
            self.postcodedb = pd.read_csv(os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_unlabelled.csv')))

        if household_file == '':
            self.households = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          'households_per_sector.csv'))
        if labelled_samples == '':
            self.labelled_samples = pd.read_csv(os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_sampled.csv')))

        self.sample_labels = sample_labels

        #self.postcodes_labled = pd.read_csv(sample_labels)

        #if we want to transform cumstom
        #num_pipe = make_column_transformer((StandardScaler(),['TempPreviousMonth']))
        #union = make_union(MonthEncoder(),CityEncoder(), CoordinatesFeatureCross(), num_pipe)
        #self.model = make_pipeline(union, LinearRegression())
        #placeholders for X_train and y_train
        self.X_train,self.X_test,self.y_train,self.y_test,self.model = {},{},{},{},{}

        self.risk_dict={1:.01,
            2:.05,
            3:.1,
            4:.5,
            5:1,
            6:1.5,
            7:2,
            8:3,
            9:4,
            10:5}
       

        return None
    def compose_X_y(self,label):

        #Think we only need to split into X and ys
        self.X_train[label] = self.labelled_samples.drop(columns=self.sample_labels) #drop all labled columns from all X_trains 
        self.y_train[label] = self.labelled_samples[label]

        self.X_train[label], self.X_test[label], self.y_train[label], self.y_test[label] = train_test_split(self.X_train[label], self.y_train[label], test_size=0.2, random_state=5)

        #convert risk classes into risk probabilites
        if label=='riskLabel':
            self.y_train[label]=self.apply_dictionary(self.y_train[label])
            self.y_test[label]=self.apply_dictionary(self.y_train[label])

        return None

    def apply_dictionary(self,y):
        """
        y is the riskLabel
        risk_dictionary is the dictionary with risk classes and probabilities
        """
        trans_y = []
        if (all(isinstance(x, int) for x in list(y)) == True): 
            for i in list(y):
                trans_y.append(self.risk_dict[i])
        else: #if the values in y are intigers (flood classes)
            for i in list(y):
                risk_label = list(self.risk_dict.keys())[list(self.risk_dict.values()).index(round(i,2))]
                trans_y.append(risk_label)

        return pd.Series(trans_y, index=y.index)

    def train(self, grid_mtrx='',score=''):
        """Train the model using a labelled set of samples.
        
        Parameters
        ----------
        
        labelled_samples : str, optional
            Filename of a .csv file containing a labelled set of samples.
        """
        report = {}    
        for label in self.sample_labels:

            self.compose_X_y(label)
            
            #pipeline for data processing, what features do we use what do we drop
            transformer = Tr(self.X_train[label])
            
            #account for the dropped columns - should drop by default
            #self.X_train[label] = transformer.data
            
            feature_processor = transformer.make_my_pipeline()
            model_pipe = Pipeline([ ('preprocessor',feature_processor),
                                    ('regressor', DummyRegressor())])

            grid_search = GridSearchCV(model_pipe, grid_mtrx, cv=5,
                           scoring=score,
                           n_jobs=-1)

            self.model[label] = grid_search.fit(self.X_train[label], self.y_train[label])

            print("**************************************************************************************")
            print(np.mean(self.y_train[label]),np.var(self.y_train[label]),np.median(self.y_train[label]),
                    np.min(self.y_train[label]),np.max(self.y_train[label]))

            print(f"***********************************{label}************************************")
            print(f"CV: {self.model[label].cv_results_}")
            print(f"Best Estimator: {self.model[label].best_estimator_}")
            print(f"Best Params: {self.model[label].best_params_}")
            print(f"Best Score: {self.model[label].best_score_}")
            #print(f"Mean Score Time: {self.model[label].mean_score_time}")
            report[label] = [ {'CV': self.model[label].cv_results_},
                              {'CV': self.model[label].best_estimator_},
                              {'CV': self.model[label].best_params_},
                              {'CV': self.model[label].best_score_}]

        return self.model


    def predict(self):

        y_pred={}
        for label in self.sample_labels:
            y_pred[label] = self.model[label].best_estimator_.predict(self.X_test[label])
            print(label)
        return None

