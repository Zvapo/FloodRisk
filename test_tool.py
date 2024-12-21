import numpy as np
import pandas as pd
import flood_tool

'''
Tests

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

UNLABELLED      ='postcodes_unlabelled.csv'# 'new_unlabelled.csv'
LABELLED        ='postcodes_sampled.csv'#'new_labelled.csv'
GROUND_TRUTH    =''#'new_truth.csv'
HOUSEHOLDS      ='households_per_sector.csv'#'new_household_per_sector.csv'

new_tool = flood_tool.Tool( postcode_file=UNLABELLED, 
                            sample_labels=['riskLabel','medianPrice'],
                            household_file=HOUSEHOLDS)
methods = new_tool.get_default_methods()
new_tool.train(labelled_samples=LABELLED, method=methods)

print("get_flood_class_from_postcodes")
POSTCODES = ['TN6 3AW','BN7 2HP','BN1 5PF']
predict = new_tool.get_flood_class_from_postcodes(POSTCODES,method=methods['riskLabel'])
print(predict)
#assert issubclass(type(predict), pd.Series)

print("get_median_house_price_estimat'BN1 5PF','BN3 7LP'e")
POSTCODES = [ 'TN6 3AW','BN7 2HP','BN1 5PF']
predict = new_tool.get_median_house_price_estimate(POSTCODES,method=methods['medianPrice'])
print(predict)
#assert issubclass(type(predict), pd.Series)

print("get_flood_class_from_OSGB36_locations")
coords = [[530401.0, 541934.0, 169999.0], [105619.0, 110957.0,208999.0]] #[169999.0],[208999.0]
predict = new_tool.get_flood_class_from_OSGB36_locations(coords[0],coords[1],method=methods['location'])
print(predict)

#assert issubclass(type(predict), pd.Series)

print("get_flood_class_from_WGS84_locations")
coords = [[56.5],[-1.54]]
predict = new_tool.get_flood_class_from_WGS84_locations(coords[0],coords[1],method=methods['location'])
print(predict)
#assert issubclass(type(predict), pd.Series)

print("get_local_authority_estimate")
predict = new_tool.get_local_authority_estimate(coords[0],coords[1],method=methods['location'])
print(predict)
#assert issubclass(type(predict), pd.Series)

print("get_easting_northing")
POSTCODES = [ 'TN6 3AW','BN7 2HP','BN1 5PF']
r = new_tool.get_easting_northing(POSTCODES)
print(r)

#assert issubclass(type(predict), pd.DataFrame)
print("get_lat_long")
r = new_tool.get_lat_long(POSTCODES)
print(r)
#assert issubclass(type(predict), pd.DataFrame)

print("get_total_value")
POSTCODES = ['TN6 3AW','BN7 2HP','BN1 5PF','KT17 4DG'] #,'SW6 7XE'
r = new_tool.get_total_value(POSTCODES)
print(r)

#assert issubclass(type(predict), pd.Series)
print("get_annual_flood_risk")
print(new_tool.get_annual_flood_risk(POSTCODES))

#assert issubclass(type(predict), pd.Series)
'''
SCORES = np.array([[100, 80, 60, 60, 30, 0, -30, -600, -1800, -2400],
                   [80, 100, 80, 90, 60, 30, 0, -300, -1200, -1800],
                   [60, 80, 100, 120, 90, 60, 30, 0,  -600, -1200],
                   [40, 60, 80,  150, 120, 90, 60, 300, 0, -600],
                   [20, 40, 60, 120, 150, 120, 90, 600, 600, 0],
                   [0, 20, 40, 90, 120, 150, 120, 900, 1200, 600],
                   [-20, 0, 20, 60, 90, 120, 150, 1200, 1800, 1200],
                   [-40, -20, 0, 30, 60, 90, 120, 1500, 2400, 1800],
                   [-60, -40, -20, 0, 30, 60, 90, 1200, 3000, 2400],
                   [-80, -60, -40, -30, 0, 30, 60, 900, 2400, 3000]])

#truth = extract from GROUND_TRUTH
print sum([SCORES[_p-1, _t-1]
     for _p, _t in zip(predicted, truth)])
'''
