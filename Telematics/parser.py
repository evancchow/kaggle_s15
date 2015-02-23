from os import listdir
import random
import csv
import numpy as np
from gen_features import *

DATA_PATH = "./data/drivers/"

def parse_data(file):
	data = []
	with open(file) as f:
		reader = csv.DictReader(f)
		for item in reader:
			data.append(np.asarray([float(item['x']), float(item['y'])]))
	return data

def new_parse(file):
	x = []
	with open(file) as f:
		reader = csv.DictReader(f)
		for item in reader:
			x.append(float(item['x']))
			y.append(float(item['y']))
	return [x,y]

def get_driver_list():
	driver_list = []
	for item in listdir(DATA_PATH):
		if not (item == ".DS_Store"):
			driver_list.append(int(item))
	return driver_list

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

	return [reference_set, reference_output, reference_drivers]

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

	return [training_set, training_output]