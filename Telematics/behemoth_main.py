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

np.set_printoptions(threshold=np.nan)


start = datetime.datetime.now()
DATA_PATH = "./data/drivers/"

MODEL_LIST = ['gbr', 'lr', 'sgd', 'en', 'lasso', 'rf']
MODEL_CHOICE = 'lr'

trip_indices = []

ctr = 1
while ctr <= 200:
	trip_indices.append(str(ctr))
	ctr +=1

trip_indices = sorted(trip_indices)
print trip_indices

random.seed(42)

def parse_data(file):
	data = []
	with open(file) as f:
		reader = csv.DictReader(f)
		for item in reader:
			data.append(np.asarray([float(item['x']), float(item['y'])]))
	return data

#From SciPy cookbook
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def new_parse(file):
	x = []
	with open(file) as f:
		reader = csv.DictReader(f)
		for item in reader:
			x.append(float(item['x']))
			y.append(float(item['y']))
	return [x,y]

def get_speed_quantiles(data):
	x_vals = zip(*data)[0]
	y_vals = zip(*data)[1]

	x_smooth = np.asarray(savitzky_golay(x_vals, 29, 3))
	y_smooth = np.asarray(savitzky_golay(y_vals, 29, 3))

	x_smooth_diff = diff(x_smooth)
	y_smooth_diff = diff(y_smooth)

	comp = x_smooth_diff ** 2 + y_smooth_diff ** 2

	num = 0.05
	nums = []
	while num <= 1:
		nums.append(num)
		num += 0.1

	quantiles = mquantiles(comp, nums)
	#print quantiles
	#time.sleep(0.1)
	return np.asarray(quantiles)

def smooth(x, y, steps):
    """
    Returns moving average using steps samples to generate the new trace

    Input: x-coordinates and y-coordinates as lists as well as an integer to indicate the size of the window (in steps)
    Output: list for smoothed x-coordinates and y-coordinates
    """
    xnew = []
    ynew = []
    for i in xrange(steps, len(x)):
        xnew.append(sum(x[i-steps:i]) / float(steps))
        ynew.append(sum(y[i-steps:i]) / float(steps))
    return xnew, ynew

def distance(x0, y0, x1, y1):
    """Computes 2D euclidean distance"""
    return hypot((x1 - x0), (y1 - y0))

#http://math.stackexchange.com/questions/1115619/centripetal-acceleration-for-a-polyline
def centripetal_accel(speed, accel):
	cent_vec = []
	length = len(speed)
	ctr = 0
	while ctr < length:
		val = speed[ctr]
		my_norm = norm(val)
		if not(my_norm == 0):
			speed_unit = val/my_norm
			cent_val = norm(np.cross(accel[ctr], speed_unit))
			cent_vec.append(cent_val)
		ctr += 1

	num = 0.1
	nums = []
	while num <= 1:
		nums.append(num)
		num += 0.1

	quantiles = mquantiles(cent_vec, nums)
	return np.asarray(quantiles)

def tangential_accel(speed, accel):
	tan_vec = []
	length = len(speed)
	ctr = 0
	while ctr < length:
		val = speed[ctr]
		my_norm = norm(val)
		if not(my_norm == 0):
			speed_unit = val/my_norm
			tan_val = np.dot(accel[ctr], speed_unit)
			tan_vec.append(tan_val)
		ctr += 1
	num = 0.1
	nums = []
	while num <= 1:
		nums.append(num)
		num += 0.1

	quantiles = mquantiles(tan_vec, nums)
	return np.asarray(quantiles)

def velocities_and_distance_covered(x, y):
    """
    Returns velocities just using difference in distance between coordinates as well as accumulated distances

    Input: x-coordinates and y-coordinates as lists
    Output: list of velocities
    """
    v = []
    distancesum = 0.0
    for i in xrange(1, len(x)):
        dist = distance(x[i-1], y[i-1], x[i], y[i])
        v.append(dist)
        distancesum += dist
    return v, distancesum

def optimized_features(data):
	pass


#Returns the average and max speeds of the path.
def get_speed_features(data):
	speeds = []
	length = len(data)
	ctr = 0
	while ctr < length-1:
		speeds.append(norm(data[ctr] - data[ctr+1]))
		ctr += 1
	return np.asarray([np.average(speeds), max(speeds)])

def get_length(data):
	return np.asarray([len(data)])

def get_distance(data):
	start = data[0]
	end = data[-1]
	return np.asarray([norm(start-end)])

def get_features(data):
	x_vals = zip(*data)[0]
	y_vals = zip(*data)[1]

	x_smooth = savitzky_golay(x_vals, 29, 3)
	y_smooth = savitzky_golay(y_vals, 29, 3)

	x_speed = savitzky_golay(x_vals, 29, 3, 1)
	y_speed = savitzky_golay(y_vals, 29, 3, 1)

	x_accel = savitzky_golay(x_vals, 29, 3, 2)
	y_accel = savitzky_golay(y_vals, 29, 3, 2)

	speed_vec = zip(x_speed, y_speed)
	accel_vec = zip(x_accel, y_accel)

	c_accel = centripetal_accel(speed_vec, accel_vec)
	t_accel = tangential_accel(speed_vec, accel_vec)

	x_len = len(x_smooth)
	ctr = 0

	smoothed_data = []
	while ctr < x_len:
		smoothed_data.append([x_smooth[ctr], y_smooth[ctr]])
		ctr += 1

	length = get_length(data)
	dist = get_distance(data)

	data = np.asarray(smoothed_data)
	speed_quantiles = get_speed_quantiles(data)
	speed_features = get_speed_features(data)

	
	#features = speed_features + length + dist + c_accel + t_accel + speed_quantiles
	features = np.concatenate((speed_features, length, dist, c_accel, t_accel, speed_quantiles), axis=0)
	return features

def get_driver_list():
	driver_list = []
	for item in listdir(DATA_PATH):
		if not (item == ".DS_Store"):
			driver_list.append(int(item))
	return driver_list

def test_if_valid(data):
	length = len(data)
	ctr = 0
	for val in data:
		if val > 0.5:
			ctr += 1
	if float(ctr)/float(length) > 0.5: 
		is_valid = True
	else:
		is_valid = False 
	return ctr, length, is_valid

def get_reference_data(num_false):
	driver_list = get_driver_list()

	reference_set = []
	reference_output = []

	global reference_drivers
	reference_drivers = []
	#reference_drivers = [1064,1597,1735,2670,2925]
	#reference_drivers = [2670, 1735, 1064, 1597]
	ctr = 0
	while ctr < num_false:
		false_driver_num = random.choice(driver_list)
		print "FD"+str(false_driver_num)
		false_path = DATA_PATH + str(false_driver_num) + "/"

		reference_drivers.append(false_driver_num)

		for file in listdir(false_path):
			data = parse_data(false_path+file)

			'''
			x = zip(*data)[0]
			y = zip(*data)[1]
			x_s, y_s = smooth(x,y,10)
			v, distancecovered = velocities_and_distance_covered(x_s,y_s)
			maxspeed = max(v)
			triplength = distance(x[0], y[0], x[-1], y[-1])
			triptime = len(x)

			features = []
			features.append(triplength)
			features.append(triptime)
			features.append(distancecovered)
			features.append(maxspeed)
			'''


			false_features = get_features(data)
			#false_features = [len(false_data)]
			#false_features = get_speed_quantiles(data)

			#print features
			#features = get_speed_quantiles(data)

			reference_set.append(false_features)
			reference_output.append(0)

		driver_list.remove(false_driver_num)
		ctr += 1

	return [reference_set, reference_output]







#Generates training set.  All data associated with a given driver is assumed to be from that driver.
#Then, we take num_false random other drivers and add data known to be from different drivers. 

def get_training_set(driver_number, num_false):
	path = DATA_PATH+str(driver_number) + "/"
	training_set = []
	training_output = []

	for file in listdir(path):
		data = parse_data(path+file)
	
		'''
		x = zip(*data)[0]
		y = zip(*data)[1]
		x_s, y_s = smooth(x,y,10)
		v, distancecovered = velocities_and_distance_covered(x_s,y_s)
		maxspeed = max(v)
		triplength = distance(x[0], y[0], x[-1], y[-1])
		triptime = len(x)

		features = []
		features.append(triplength)
		features.append(triptime)
		features.append(distancecovered)
		features.append(maxspeed)
		'''

		features = get_features(data)
		#features = [len(data)]
		#time.sleep(3)
		#features = get_speed_quantiles(data)
		training_set.append(features)
		training_output.append(1)

	driver_list = get_driver_list()
	driver_list.remove(driver_number)

	'''
	while ctr < num_false:
		false_driver_num = random.choice(driver_list)
		print "FD"+str(false_driver_num)
		false_path = DATA_PATH + str(false_driver_num) + "/"

		for file in listdir(false_path):
			false_data = parse_data(false_path+file)
			#false_features = get_features(false_data)
			#false_features = [len(false_data)]
			false_features = get_speed_quantiles(false_data)
			training_set.append(false_features)
			training_output.append(0)

		driver_list.remove(false_driver_num)
		ctr += 1
	'''

	return training_set, training_output

def train(driver_number, reference, num_false):
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
		model.fit(ts_added, ts_output_added)
		predictions[str(driver_number)] = (model.predict(ts_added[:200]))
		#ts_scaled = (preprocessing.scale(ts[0]), ts[1])

	return predictions
	

def generate_submission_file(num_false):

	driver_list = get_driver_list()
	reference = get_reference_data(num_false)

	driver_list = [1]
	print len(driver_list)
	
	'''
	for driver_num in driver_list:
		train(driver_num, reference, num_false)
	'''

	predictions = Parallel(n_jobs=20)(delayed(train)(driver_num, reference, num_false) for driver_num in driver_list)
	predictions = reduce(lambda x,y: dict(x.items() + y.items()), predictions)

	#print predictions
	#print predictions['1']
	#print predictions['2']
	'''
	try:
		print predictions['1']
		print test_if_valid(predictions['1'])
	'''
	with open('test_submission.csv', 'w') as csvfile:
		fieldnames = ['driver_trip', 'prob']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for driver_number in driver_list:
			print "Writing:"+str(driver_number)
			ctr = 0
			while ctr < 200:
				writer.writerow({'driver_trip': str(driver_number)+'_'+trip_indices[ctr], 'prob':predictions[str(driver_number)][ctr]})
				ctr += 1

generate_submission_file(2)
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


