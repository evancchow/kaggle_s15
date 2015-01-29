import csv
import numpy as np 
from numpy.linalg import norm
import os
from os import listdir
import random
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor


DATA_PATH = "./data/drivers/"

def parse_data(file):
	data = []
	with open(file) as f:
		reader = csv.DictReader(f)
		for item in reader:
			data.append(np.asarray([float(item['x']), float(item['y'])]))
	return data

#Returns the average speed of the path.
def get_speed(data):
	speeds = []
	length = len(data)
	ctr = 0
	while ctr < length-1:
		speeds.append(norm(data[ctr] - data[ctr+1]))
		ctr += 1
	return np.average(speeds)

def get_driver_list():
	driver_list = []
	for item in listdir(DATA_PATH):
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
		speed = get_speed(data)
		training_set.append([speed])
		training_output.append(1)

	driver_list = get_driver_list()
	driver_list.remove(driver_number)

	ctr = 0
	while ctr < num_false:
		false_driver_num = random.choice(driver_list)
		false_path = DATA_PATH + str(false_driver_num) + "/"

		for file in listdir(false_path):
			false_data = parse_data(path+file)
			false_speed = get_speed(false_data)
			training_set.append([false_speed])
			training_output.append(0)

		driver_list.remove(false_driver_num)
		ctr += 1

	return training_set, training_output

ts = get_training_set(1,5)
#A natural first step is to implement a logistic regressor, since we want to output probability.
#model = LogisticRegression()
#model = GradientBoostingRegressor()
model = SGDRegressor()
model.fit(ts[0], ts[1])
prediction = model.predict(ts[0])
print prediction[:-10]
print prediction[:20]


