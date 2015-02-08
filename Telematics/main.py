import csv
import numpy as np 
from numpy.linalg import norm
import os
from os import listdir
import random
from sklearn.linear_model import LogisticRegression, SGDRegressor, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import datetime

start = datetime.datetime.now()
DATA_PATH = "./data/drivers/"

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

#Returns the average and max speeds of the path.
def get_speed_features(data):
	speeds = []
	length = len(data)
	ctr = 0
	while ctr < length-1:
		speeds.append(norm(data[ctr] - data[ctr+1]))
		ctr += 1
	return [np.average(speeds), max(speeds)]

def get_length(data):
	return [len(data)]

def get_distance(data):
	start = data[0]
	end = data[-1]
	return [norm(start-end)]

def get_features(data):
	x_vals = zip(*data)[0]
	y_vals = zip(*data)[1]

	x_smooth = savitzky_golay(x_vals, 29, 3)
	y_smooth = savitzky_golay(y_vals, 29, 3)

	x_len = len(x_smooth)
	ctr = 0

	smoothed_data = []
	while ctr < x_len:
		smoothed_data.append([x_smooth[ctr], y_smooth[ctr]])
		ctr += 1

	data = np.asarray(smoothed_data)

	speed_features = get_speed_features(data)
	length = get_length(data)
	dist = get_distance(data)
	features = speed_features + length + dist
	return features

def get_driver_list():
	driver_list = []
	for item in listdir(DATA_PATH):
		if not (item == ".DS_Store"):
			driver_list.append(int(item))
	return driver_list

#Generates training set.  All data associated with a given driver is assumed to be from that driver.
#Then, we take num_false random other drivers and add data known to be from different drivers. 

def get_training_set(driver_number, num_false):
	path = DATA_PATH+str(driver_number) + "/"
	training_set = []
	training_output = []

	for file in listdir(path):
		data = parse_data(path+file)
		features = get_features(data)
		#features = [len(data)]
		training_set.append(features)
		training_output.append(1)

	driver_list = get_driver_list()
	driver_list.remove(driver_number)

	ctr = 0
	while ctr < num_false:
		false_driver_num = random.choice(driver_list)
		print "FD"+str(false_driver_num)
		false_path = DATA_PATH + str(false_driver_num) + "/"

		for file in listdir(false_path):
			false_data = parse_data(false_path+file)
			false_features = get_features(false_data)
			#false_features = [len(false_data)]
			training_set.append(false_features)
			training_output.append(0)

		driver_list.remove(false_driver_num)
		ctr += 1

	return training_set, training_output

def generate_submission_file(num_false):
	driver_list = get_driver_list()
	print len(driver_list)
	predictions = {}
	#driver_list = [1]
	for driver_number in driver_list:
		print "GB training on:"+str(driver_number)
		ts = get_training_set(driver_number,num_false)
		f = open('ts.txt','wb')
		f.write(str(ts))

		'''
		path = DATA_PATH+str(driver_number) + "/"
		for file in listdir(path):
			data = parse_data(path+file)
			features = get_features(data)
			training_set.append(features)
			training_output.append(1)
		'''

		model = GradientBoostingRegressor()
		#model = LogisticRegression()
		#model = SGDRegressor()
		#model = Lasso()
		#model = ElasticNet()
		#ts_scaled = (preprocessing.scale(ts[0]), ts[1])
		model.fit(ts[0], ts[1])
		predictions[str(driver_number)] = (model.predict(ts[0][:200]))

	print predictions
	with open('submission_en.csv', 'w') as csvfile:
		fieldnames = ['driver_trip', 'prob']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for driver_number in driver_list:
			print "Writing:"+str(driver_number)
			ctr = 0
			while ctr < 200:
				writer.writerow({'driver_trip': str(driver_number)+'_'+str(ctr+1), 'prob':predictions[str(driver_number)][ctr]})
				ctr += 1

generate_submission_file(5)
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


