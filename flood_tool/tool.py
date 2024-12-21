"""Tool Module"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor,SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


from .transformer import Tr
from .geo import GeoTransformer


__all__ = ['Tool']

### Helper functions

def _postcode_normalize(postcodes):
    '''
    Normalize input postcodes. Example: ' sW7 2az  ' -> 'SW7 2AZ'
    
    Parameters
    ----------
    postcode: sequence of str
        The input postcodes to normalize
        
    Returns
    ----------
    normalized_postcodes: array of str
    '''
    normalized_postcodes = []
    for postcode in np.array(postcodes, ndmin=1):
        postcode = str(postcode)
        postcode = postcode.upper()
        postcode = postcode.strip()
        inward = postcode[-3:]
        if not (inward[0].isnumeric() and inward[1:].isalpha()):
            warnings.warn('Postcode '+postcode+' is invalid. This will result in a NaN in the final outputs.')
            normalized_postcodes.append(postcode)
            continue
        outward = postcode[:-3]
        outward = outward.strip()
        if len(outward)<2 or len(outward)>4 or not outward[0].isalpha():
            warnings.warn('Postcode '+postcode+' is invalid. This will result in a NaN in the final outputs.')
            normalized_postcodes.append(postcode)
            continue
        normalized_postcodes.append(outward+' '+inward)
    return np.array(normalized_postcodes)


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='', sample_labels=['riskLabel','medianPrice'],
                 household_file=''):
        """
        Parameters
        ----------

        full_postcode_file : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        household_file : str, optional
            Filename of a .csv file containing information on households
            by postcode.

        sample_lables : str, optional
            string or an array of lables  
        """

        full_postcode_file = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         ('postcodes_unlabelled.csv' if postcode_file == '' else postcode_file)))


        household_file = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          ('households_per_sector.csv' if household_file == '' else household_file)))

        self.sample_labels = sample_labels
            
        self.postcodedb = pd.read_csv(full_postcode_file)
        self.households = pd.read_csv(household_file)
        
        #placeholders for X_train and y_train
        self.X_train,self.y_train,self.model = {},{},{}
        self.labelled_samples = ''
        
        self.risk_dict_bound = {10:(5,4),
                                9:(4,3),
                                8:(3,2),
                                7:(2,1.5),
                                6:(1.5,1),
                                5:(1,0.5),
                                4:(0.5,0.1),
                                3:(0.1,0.05),
                                2:(0.05,0.01),
                                1:(0.01, 0)}
       
        #best models to be used by default
        self.methods = {'riskLabel': self.get_flood_class_from_postcodes_methods()['KNNR'],
                        'medianPrice': self.get_house_price_methods()['SGDR'],
                        'location': self.get_flood_class_from_locations_methods()['KNNR']}
        return None

    def get_default_methods(self):
        """
        Function to obtain defauld methods from outside of the class

        Returns:
            self.methods 
        """
        return self.methods

    def compose_X_y(self,label):

        #Think we only need to split into X and ys
        self.X_train[label] = self.labelled_samples.drop(columns=self.sample_labels) #drop all labled columns from all X_trains 
        self.y_train[label] = self.labelled_samples[label]

        #convert risk classes into risk probabilites
        if label=='riskLabel':
            self.y_train[label]=self.apply_dictionary(self.y_train[label])

        return None

    def train(self, labelled_samples='', method=0):
        """Train the model using a labelled set of samples.
        
        Parameters
        ----------
        
        labelled_samples : str, optional
            Filename of a .csv file containing a labelled set of samples.
        """

        #make use of labeled samples
        self.labelled_samples = pd.read_csv(os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         ('postcodes_sampled.csv' if labelled_samples == '' else labelled_samples))))

        if (method == 0):
            method = {  'riskLabel': 'all_zero_risk',
                        'medianPrice': 'all_england_median'
                    }
            
        for label in self.sample_labels:
            self.compose_X_y(label)
            
            #pipeline for data processing, what features do we use what do we drop
            transformer = Tr(self.X_train[label])
            
            #account for the dropped columns - should drop by default
            #self.X_train[label] = transformer.data
            
            feature_processor = transformer.make_my_pipeline()
            model_pipe = Pipeline([ ('preprocessor',feature_processor),
                                    ('regressor', self.methods[label])])
            self.model[label] = model_pipe.fit(self.X_train[label], self.y_train[label])

        return self.model


    def apply_dictionary(self,y):
        """
        y is the riskLabel
        risk_dictionary is the dictionary translating between risk classes and probabilities
        """
        trans_y = []
        if all(isinstance(x, int) for x in list(y)) == True: 
            for i in list(y):
                trans_y.append(self.risk_dict_bound[i][0])
            return pd.Series(trans_y)
        
        elif all(isinstance(x, float) for x in list(y)) == True: #if the values in y are intigers (flood classes)
            for i in y:
                for j in self.risk_dict_bound.values():
                    if (j[1] <= i <= j[0]):
                        class_ = list(self.risk_dict_bound.keys())[list(self.risk_dict_bound.values()).index(j)]
                        trans_y.append(class_)
                        break
                    elif i > 5.0:
                        trans_y.append(10)
                        break
                    else:
                        pass
        return pd.Series(trans_y, index=y.index)

    def validate_postcodes(self, postcodes):

        frame = self.postcodedb.copy()
        frame = frame.set_index('postcode') #make postcode as an index
        valid_postcodes = []
        invalid_postcodes = []

        for i in range(len(postcodes)):
            if postcodes[i] in frame.index:
                valid_postcodes.append(postcodes[i])
            else:
                invalid_postcodes.append(postcodes[i])
        return valid_postcodes,invalid_postcodes

    def get_postcodes_from_easting_northing(self, eastings, northings):
        """
        Select nearest postcode given OSGB36 coordinates

        Args:
            dataframe with coordinates

        ----------
        Returns 
            Series with postcodes
        """

        X1 = self.labelled_samples[['easting', 'northing']]
        X2 = self.postcodedb[['easting', 'northing']]
        y1 = self.labelled_samples[['postcode']]
        y2 = self.postcodedb[['postcode']]
        y = [y1, y2]
        y = np.ravel(pd.concat(y))
        X = [X1, X2]
        X = pd.concat(X)

        # encode postcodes
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        # fit knn model
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, y)
        X_unlabelled = {'easting': eastings, 'northing': northings}

        X_unlabelled = pd.DataFrame(data=X_unlabelled)
        # make prediction 
        y_pred = le.inverse_transform(knn.predict(X_unlabelled))
        return pd.Series(data=y_pred,index=[eastings,northings])

    def get_easting_northing(self, postcodes):
        """Get a frame of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NaN.
         """
        
        postcodes = _postcode_normalize(postcodes)

        frame = self.postcodedb.copy()
        frame = frame.set_index('postcode') #make postcode as an index
        
        #capture the invalid postcode
        valid_postcodes, invalid_postcodes = self.validate_postcodes(postcodes)
        

        result = frame.loc[valid_postcodes, ['easting', 'northing']]
        
        #returning invalid postcode as nan
        num = {'easting': [np.nan],
              'northing': [np.nan]}
        return_nan = pd.DataFrame(num,index = invalid_postcodes)
        
        result = result.append(return_nan)
        
        return result



    def get_lat_long(self, postcodes):
        """Get a frame containing GPS latitude and longitude information for a
        collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NAN.
        """


        frame = self.get_easting_northing(postcodes)
        
        #capture the invalid postcode
        invalid_postcodes = [index for index, row in frame.iterrows() if row.isnull().any()]

        frame = frame.dropna() 
        frame['latitude'],frame['longitude'] = GeoTransformer.get_gps_lat_long_from_easting_northing(frame.easting,frame.northing)
        frame.drop(['easting','northing'],axis=1,inplace =True)
 
        #returning invalid postcode as nan
        num = {'latitude': [np.nan],
              'longitude': [np.nan]}
        return_nan = pd.DataFrame(num,index = invalid_postcodes)
        
        frame = frame.astype(np.float64)
        result = frame.append(return_nan)

        return result

    @staticmethod
    def get_flood_class_from_postcodes_methods():
        """
        Get a dictionary of available flood probablity classification methods
        for postcodes.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_postcode method.
        """

        grid = {        'all_zero_risk': 0,
                        'grdboost':GradientBoostingRegressor(learning_rate = 0.05, n_estimators = 1000, loss = 'squared_error', min_samples_split = 4),
                        'KNNR':KNeighborsRegressor(n_neighbors = 8, p = 2, weights = 'distance')
            }
        return grid

    def get_flood_class_from_postcodes(self, postcodes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            get_flood_class_from_postcodes_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """

        if (method == 0):
            return pd.Series(data=np.ones(len(postcodes), int),
                    index=np.asarray(postcodes),
                    name='riskLabel')
        elif (self.postcodedb[self.postcodedb['postcode'].isin(postcodes)].empty):
            return pd.DataFrame()
        else:
            pred = self.model['riskLabel'].predict(self.postcodedb[self.postcodedb['postcode'].isin(postcodes)])
            #adjust in case a postcode hadn't been found
            postcodes = self.postcodedb[self.postcodedb['postcode'].isin(postcodes)].postcode
            pred = self.apply_dictionary(pd.Series(data=pred,
                            index = np.asarray(postcodes),
                            name = 'riskLabel'))

            return pred
            

    @staticmethod
    def get_flood_class_from_locations_methods():
        """
        Get a dictionary of available flood probablity classification methods
        for locations.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_OSGB36_locations and
             get_flood_class_from_OSGB36_locations method.
        """

        grid = {        'all_zero_risk': 0,
                        'grdboost':GradientBoostingRegressor(learning_rate = 0.05, n_estimators = 1000, loss = 'squared_error', min_samples_split = 4),
                        'KNNR':KNeighborsRegressor(n_neighbors = 8, p = 2, weights = 'distance')
            }
        return grid

    def get_flood_class_from_OSGB36_locations(self, eastings, northings, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of OSGB36_locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """
       
        if (method == 0):
            return pd.Series(data=np.ones(len(eastings), int),
                             index=[(est, nth) for est, nth in
                                    zip(eastings, northings)],
                             name='riskLabel')
        elif method == 1 :
            return f"Invalid method"
        else:
            #estimate nearest postcode
            postcodes = self.get_postcodes_from_easting_northing(eastings,northings)
            
            #if we have data get risk
            df1 = self.labelled_samples[self.labelled_samples['postcode'].isin(postcodes.values)]
            tmp1_=pd.Series(df1['riskLabel'].values,index=df1['postcode'])
                
            #if we don't - predict
            tmp2_ = self.get_flood_class_from_postcodes(postcodes.values,self.methods['riskLabel'])
            
            if (tmp2_.empty):
                return (f"Postcode(s) not found" if tmp1_.empty else tmp1_)
            elif (tmp1_.empty):
                return (f"Postcode(s) not found" if tmp2_.empty else tmp2_)
            else:
            #combine
                predict = pd.concat([tmp1_,tmp2_])
                postcodes = pd.Series(dict((v,k) for k,v in postcodes.iteritems()))
                predict = pd.concat([postcodes,predict],axis=1)
                return predict.set_index(0)


    def get_flood_class_from_WGS84_locations(self, longitudes, latitudes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        if (method == 0):
            return pd.Series(data=np.ones(len(longitudes), int),
                             index=[(lng, lat) for lng, lat in
                                    zip(longitudes, latitudes)],
                             name='riskLabel')
        else:
            _ = GeoTransformer.get_gps_lat_long_from_easting_northing(longitudes, latitudes)
            result = self.get_flood_class_from_OSGB36_locations(_[0],_[1],self.methods['location'])

            return  pd.Series(data=result.values,
                    index=[longitudes, latitudes],
                    name='riskLabel')

    @staticmethod
    def get_house_price_methods():
        """
        Get a dictionary of available flood house price regression methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_median_house_price_estimate method.
        """
        
        grid = {        'all_england_median': 0,
                        'SGDR': SGDRegressor(max_iter = 13000, penalty = 'l1', alpha = 0.05)
        }
        return grid

    def get_median_house_price_estimate(self, postcodes, method=0):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_house_price_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """

        if (method == 0):
            return pd.Series(data=np.full(len(postcodes), 245000.0),
                             index=np.asarray(postcodes),
                             name='medianPrice')
        else:
            return pd.Series(data = self.model['medianPrice'].predict(self.postcodedb[self.postcodedb['postcode'].isin(postcodes)]),
                            index = np.asarray(postcodes),
                            name = 'medianPrice')

    @staticmethod
    def get_local_authority_methods():
        """
        Get a dictionary of available local authorithy classification methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_altitude_estimate method.
        """
        return {'Do Nothing': 0}

    def get_local_authority_estimate(self, eastings, northings, method=0):
        """
        Generate series predicting local authorities for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in
            self.get_altitude_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of authorities indexed by eastings and northings.
        """

        if (method == 0):
            return pd.Series(data=np.full(len(eastings), 'Unknown'),
                             index=[(est, nth) for est, nth in
                                    zip(eastings, northings)],
                             name='localAuthority')
        else:
            X = self.postcodedb[['easting', 'northing']]
            y = np.ravel(self.postcodedb[['localAuthority']])
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X, y)
            X_unlabelled = {'easting': eastings, 'northing': northings}
          
            X_unlabelled = pd.DataFrame(data=X_unlabelled)
            y_pred = knn.predict(X_unlabelled)

            #get postcodes 
            return pd.Series(data = le.inverse_transform(y_pred), index=[eastings,northings])


    def get_total_value(self, postal_data):
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units.


        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
    
        """
        if len(postal_data[0]) <= 5:

            print('Please enter full Postcode')

        

        else:
        
            # assumes input is made up of postcodes
            # get sector from postcode 
        
            postcode_prices = pd.Series(data = self.model['medianPrice'].predict(self.postcodedb[self.postcodedb['postcode'].isin(postal_data)]),
                            index = np.asarray(postal_data), name = 'postcode_prices')
            sector_of_postcodes = pd.Series(data = self.postcodedb[self.postcodedb['postcode'].isin(postal_data)].sector.values,
                                             index = np.asarray(postal_data))

            #fix sectors to match with the households file format
            self.households['postcode sector'] = self.households['postcode sector'].str.replace('  ',' ')
            
            # get average no of postcodes, households per sector
            tmp_ = self.households.set_index('postcode sector')

            postcodes_per_sector = pd.Series(data = tmp_['number of postcode units'])
            households_per_sector = pd.Series(data = tmp_['households'])
            
            #get average number of households per postcode
            households_per_postcode = households_per_sector.divide(postcodes_per_sector)
            households_per_postcode = pd.Series(data = households_per_postcode[sector_of_postcodes])

            a = postcode_prices
            a = a.reset_index()
            a.index = postcode_prices.index.str[:-2]
            a.postcode_prices = a.postcode_prices.multiply(households_per_postcode)

            # multiply no. of households per postcodes by our estimated price per postcode
            postcode_prices = pd.Series(data = a.postcode_prices,
                                        index = np.asarray(a.index))

            return postcode_prices
    

    def get_annual_flood_risk(self, postcodes='',  risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        new_tool.get_annual_flood_risk(risk_labels=True)
        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        
        if(risk_labels):
            
            cost = self.get_total_value(risk_labels.index).values
            flood_probability = self.apply_dictionary(risk_labels).values
            annual_flood_risk = pd.Series(data = 0.05*cost*risk_labels, index=risk_labels.index)
            return annual_flood_risk
        
        else:
            cost = self.get_total_value(postcodes).values
            risk_labels = self.get_flood_class_from_postcodes(postcodes, self.methods['riskLabel'])
            flood_probability = self.apply_dictionary(risk_labels).values
            annual_flood_risk = pd.Series(data = 0.05*cost*flood_probability, index = postcodes)
            return annual_flood_risk

'''
Usage in flood software 

Supplying custom files: 
UNLABELLED      ='new_unlabelled.csv'
LABELLED        ='new_labelled.csv'
GROUND_TRUTH    ='new_truth.csv'
HOUSEHOLDS      ='new_household_per_sector.csv'

Initiating the class 
new_tool = flood_tool.Tool( postcode_file=UNLABELLED, 
                            sample_labels=['riskLabel','medianPrice'],
                            household_file=HOUSEHOLDS)
methods = new_tool.get_default_methods()
new_tool.train(labelled_samples=LABELLED, method=methods)

'''

