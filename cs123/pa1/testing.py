import csv
import re

FILEPATH = "visitor_log.csv"

def get_dict():
	name_dict = {}
	count = 0

	reader = csv.reader(open(FILEPATH))

	for row in reader:
		if row[0] != "NAMELAST":
			name = " ".join([row[1], row[0]]).title()
			name_dict[name] = name_dict.get(name, 0) + 1

			count += 1

	return name_dict

def visit_ten_times(name_dict):
	log = []
	for name, count in name_dict.items():
		if count >= 10:
			log.append(name)
	return log

def get_visit_dict():
    name_dict = {}
    
    reader = csv.reader(open(FILEPATH))

    for row in reader:
        ## Ignore header
        if row[19] == "visitee_namelast":
            continue
        ## Ignore single-name anomalies
        if row[19] == "" or row[20] == "":
            continue
        ## Ignore Visitor's Office anomaly
        if row[19].lower().strip() == "office" and row[20].lower().strip() == "visitors":
            continue

        name = " ".join([row[20], row[19]]).title()
        #name = row[20] + " " + row[19]
        name_dict[name] = name_dict.get(name, 0) + 1

    return name_dict

def get_year_dict():
	name_dict = {}

	##date_form = re.compile(r"[\w']+")
	date_form = re.compile(r"[0-9]{4}")

	reader = csv.reader(open(FILEPATH))

	for row in reader:
		if row[0] != "NAMELAST" and row[11].strip() != "":
			name = " ".join(filter(lambda x: x != "", [row[1], row[2], row[0]])).title()
			date = date_form.findall(row[11])
			#print(row[11])
			##print(date)
			if len(date) == 0:
				print(row[11])
			name_dict.setdefault(name, []).append(date[0])

	return name_dict

def get_yearly_visitors(name_dict):
	l = []
	for name, years in name_dict.items():
		if len(set(years)) == 2:
			l.append("{}\t{}".format(name, set(years)))
	return l


## with MI, 1624 same visitors/visitee_namelast
## ommitting MI. 2363 same