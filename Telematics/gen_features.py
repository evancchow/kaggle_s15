from smoothing import *
from accel import *
from numpy import diff

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

def distance(x0, y0, x1, y1):
    """Computes 2D euclidean distance"""
    return hypot((x1 - x0), (y1 - y0))



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