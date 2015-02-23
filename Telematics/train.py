from sklearn.linear_model import LogisticRegression, SGDRegressor, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import preprocessing, cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
from numpy import mean
import pdb

import time

from scipy import interp
from parser import *



MODEL_LIST = ['gbr', 'lr', 'sgd', 'en', 'lasso', 'rf']
MODEL_CHOICE = 'gbr'

do_cv = False

def train_main(driver_number, reference, num_false):
	reference_drivers = reference[2]

	predictions = {}
	print MODEL_CHOICE+" training on:"+str(driver_number)
	ts = get_training_set(driver_number,num_false)

	#print reference_drivers
	#print driver_number
	#print type(driver_number)

	if driver_number in reference_drivers:
		print "removed driver"
		ind = reference_drivers.index(driver_number)
		cur_data = reference
		cur_data[0][ind*50:(ind*50+200)] = []
		cur_data[1][ind*50:(ind*50+200)] = []
		print "removed driver"
	else:
		cur_data = reference

	#print ts[0]
	#print ts[1]

	ts_added = np.asarray(ts[0] + cur_data[0])
	#pdb.set_trace()
	ts_output_added = np.asarray(ts[1] + cur_data[1])

	#print np.asarray(ts[0])
	#print np.asarray(ts[0])
	#print ts_added
	#print ts_added
	#print ts_added
	#print ts_output_added
	#print ts_output_added

	#time.sleep(60)
	#f = open('ts.txt','wb')
	#f.write(str(ts))

	
	'''
	path = DATA_PATH+str(driver_number) + "/"
	for file in listdir(path):
		data = parse_data(path+file)
		features = get_features(data)
		training_set.append(features)
		training_output.append(1)
	'''

	if MODEL_CHOICE == 'lr':
		model = LogisticRegression()
		model.fit(ts_added, ts_output_added)
		LR_list = (model.predict_proba(ts_added[:200]))
		predictions[str(driver_number)] = [item[0] for item in LR_list]
	elif MODEL_CHOICE == 'gbr':
		model = GradientBoostingRegressor(n_estimators = 100, max_depth=4)
	elif MODEL_CHOICE == 'sgd':
		model = SGDRegressor()
	elif MODEL_CHOICE == 'lasso':
		model = Lasso()
	elif MODEL_CHOICE == 'en':
		model = ElasticNet()
	elif MODEL_CHOICE == 'rf':
		model = RandomForestRegressor()

	if not(MODEL_CHOICE == 'lr'):
		if do_cv:
			length = len(ts[0])
			print 'length:'+str(length)
			cv = KFold(length, 5)

			mean_tpr = 0.0
			mean_fpr = np.linspace(0, 1, 100)
			all_tpr = []

			ts_in = np.asarray(ts[0])
			ref_data_in = np.asarray(cur_data[0])

			ts_out = np.asarray(ts[1])
			ref_data_out = np.asarray(cur_data[1])

			#ts = np.asarray(ts)
			#cur_data = np.asarray(cur_data)

			ctr = 0
			while ctr < len(cur_data):
				cur_data[ctr] = np.asarray(cur_data[ctr])
				ctr += 1

			roc_aucs = np.asarray([])

			for i, (train, test) in enumerate(cv):
				train = np.asarray(train)
				test = np.asarray(test)

				'''
				ts[0][train] = np.array(ts[0][train])
				cur_data[0]
				ctr = 0
				while ctr < len(cur_data[0][train]):
					cur_data[0][train][ctr] = np.asarray(cur_data[0][train][ctr])
					ctr += 1

				ctr = 0
				while ctr < len(ts[0][train])
					ts[0][train][ctr] = np.asarray(ts[0][train][ctr])
					ctr += 1
				'''

				model.fit(np.concatenate((ts_in[train],ref_data_in[train])), np.concatenate((ts_out[train],ref_data_out[train])))
				#print ts_added[train]
				#print ts_output_added[train]
				preds = model.predict(np.concatenate((ts_in[test],ref_data_in[test])))
				fpr, tpr, thresholds = roc_curve(np.concatenate((ts_out[test],ref_data_out[test])), preds)
				#print fpr, tpr
				#mean_tpr += interp(mean_fpr, fpr, tpr)
				#mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				roc_aucs = np.append(roc_aucs, roc_auc)
			return mean(roc_aucs)


		else:
			model.fit(ts_added, ts_output_added)
			predictions[str(driver_number)] = (model.predict(ts_added[:200]))
		#ts_scaled = (preprocessing.scale(ts[0]), ts[1])

	return predictions
	