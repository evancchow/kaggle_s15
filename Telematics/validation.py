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