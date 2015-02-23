import numpy as np
from numpy.linalg import norm
from scipy.stats.mstats import mquantiles

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