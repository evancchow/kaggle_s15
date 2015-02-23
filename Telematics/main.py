import csv
import numpy as np 
from numpy.linalg import norm
from math import hypot
import os
from os import listdir
import random
from sklearn.linear_model import LogisticRegression, SGDRegressor, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import preprocessing
from scipy.stats.mstats import mquantiles
from numpy import diff
from joblib import Parallel, delayed
import time
import datetime

from smoothing import *
from train import *
from validation import *
from gen_features import *
from accel import *
from parser import *
from smoothing import *

np.set_printoptions(threshold=np.nan)

start = datetime.datetime.now()

trip_indices = []

ctr = 1
while ctr <= 200:
	trip_indices.append(str(ctr))
	ctr +=1

trip_indices = sorted(trip_indices)
#print trip_indices

FEATURE_CHOICES = ['speed']
FEATURE_CHOICE = ['speed']

random.seed(42)

def generate_submission_file(num_false):

	driver_list = get_driver_list()
	reference = get_reference_data(num_false)

	#driver_list = [1]
	#print len(driver_list)
	
	rocs = []
	#CV code
	'''
	for driver_num in driver_list:
		roc = train_main(driver_num, reference, num_false)
		rocs.append(roc)
	with open('CV_log.txt', 'w') as f:
		f.write(np.mean(rocs))
		f.write('\n')
		f.write(driver_list)
		f.write('\n')
		f.write(FEATURE_CHOICE)
	'''
	
	predictions = Parallel(n_jobs=20)(delayed(train_main)(driver_num, reference, num_false) for driver_num in driver_list)
	predictions = reduce(lambda x,y: dict(x.items() + y.items()), predictions)

	#print predictions
	#print predictions['1']
	#print predictions['2']
	'''
	try:
		print predictions['1']
		print test_if_valid(predictions['1'])
	'''
	time = datetime.datetime.now()
	with open('submissions/'+str(time)+'submission.csv', 'w') as csvfile:
		fieldnames = ['driver_trip', 'prob']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for driver_number in driver_list:
			print "Writing:"+str(driver_number)
			ctr = 0
			while ctr < 200:
				writer.writerow({'driver_trip': str(driver_number)+'_'+trip_indices[ctr], 'prob':predictions[str(driver_number)][ctr]})
				ctr += 1

generate_submission_file(1)
print datetime.datetime.now() - start
'''
ts = get_training_set(1,5)
#A natural first step is to implement a logistic regressor, since we want to output probability.
#model = LogisticRegression()
model = GradientBoostingRegressor()
#model = SGDRegressor()
model.fit(ts[0], ts[1])
prediction = model.predict(ts[0])
print prediction[:-10]
print prediction[:20]
'''


