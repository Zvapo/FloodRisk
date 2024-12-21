from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


__all__ = ['Tr']

class Tr(object):
    """test"""

    def __init__(self, data):
        self.data = data.drop(data[['postcode','localAuthority']], axis = 1)
        return None
    
    def make_my_pipeline(self):
        num_cols = self.data.select_dtypes(include=np.number).columns
        cat_cols = self.data.select_dtypes(exclude=np.number).columns

        categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ])
        numeric_transformer = Pipeline([
                ('scaler', MinMaxScaler())
            ])

        preproc = ColumnTransformer([
                ('categoricals', categorical_transformer, cat_cols),
                ('numericals', numeric_transformer, num_cols)
            ],remainder = 'drop')
        
        return preproc
    
'''
usage(an example):
import flood_tool.transformer as Tr
transformer = Tr(self.X_train[label])
transformer.make_my_pipeline()
'''