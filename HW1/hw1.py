import numpy as np
import pandas as p
import matplotlib as plt


filename = "mock_student_data.csv"

def read_data(filename):
	array = p.read_csv(filename, usecols = range(1, 9))

	mean = array.mean()
	median = array.median()
	mode = array.mode()
	standard_deviation = array.std()

	array.hist()