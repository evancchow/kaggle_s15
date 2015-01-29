import csv
def parse_data(file):
	data = []
	with open(file) as f:
		reader = csv.DictReader(f)
		for item in reader:
			data.append(item)
	return data

data = parse_data("./data/drivers/1/1.csv")
print data[:5]

