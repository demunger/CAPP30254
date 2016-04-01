#import numpy as np
import pandas as p
##import matplotlib as plt
##from genderize import Genderize
import requests
import json

def read_data(filename):
    array = p.read_csv(filename, usecols = range(1, 9))

    mean = array.mean()
    print("Mean:\n{}\n".format(mean.to_string()))

    median = array.median()
    print("Median:\n{}\n".format(median.to_string()))

    mode = array.mode()
    print("Mode:\n{}\n".format(mode.fillna("").to_string()))

    standard_deviation = array.std()
    print("Standard Deviation\n{}\n".format(standard_deviation.to_string()))

    missing_values = array.isnull().sum()
    print("Missing Values\n{}\n".format(missing_values.to_string()))

    array.hist(figsize = (15, 10))

    return array

def get_genders(array):
	array = array.where((p.notnull(array)), None)
	for index, row in array.iterrows():
		if row["Gender"] == None:
			request = requests.get("http://api.genderize.io?&name=" + row["First_name"])
			row["Gender"] = json.loads(request.text)["gender"].title()
	return array


if __name__ == '__main__':
    filename = "mock_student_data.csv"
    array = read_data(filename)
    #get_genders(array)