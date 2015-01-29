import csv
import numpy as np 
from numpy.linalg import norm

def parse_data(file):
	data = []
	with open(file) as f:
		reader = csv.DictReader(f)
		for item in reader:
			data.append(np.asarray([float(item['x']), float(item['y'])]))
	return data

def get_speed(data):
	speeds = []
	length = len(data)
	ctr = 0
	while ctr < length-1:
		speeds.append(norm(data[ctr] - data[ctr+1]))
		ctr += 1
	return np.average(speeds)




data = parse_data("./data/drivers/1/1.csv")
data2 = parse_data("./data/drivers/2/1.csv")
print data[:5]

print get_speed(data)
print get_speed(data2)
