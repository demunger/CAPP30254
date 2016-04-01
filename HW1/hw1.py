#import numpy as np
import pandas as p
##import matplotlib as plt
import requests
import json

def read_data(filename):
    return p.read_csv(filename, usecols = range(1, 9))


def summarize_data(array):
    mean = array.mean().to_string(float_format = "{:.1f}".format)
    print("Mean:\n" + mean + "\n")

    median = array.median().to_string()
    print("Median:\n" + median + "\n")

    mode = array.mode().to_string(na_rep = "", index = False)
    print("Mode:\n" + mode + "\n")

    std_deviation = array.std().to_string(float_format = "{:.2f}".format)
    print("Standard Deviation:\n" + std_deviation + "\n")

    missing_values = array.isnull().sum().to_string()
    print("Missing Values:\n" + missing_values + "\n")


def graph_data(array):
    array.hist(figsize = (15, 10))

    ##return array.where((p.notnull(array)), None)

def get_genders(array):
    for index, row in array.iterrows():
        if row["Gender"] == None:
            request = requests.get("http://api.genderize.io?&name=" + row["First_name"])
            print(json.loads(request.text))
            temp = json.loads(request.text)["gender"].title()
            print(temp)
            row["Gender"] = temp
    return array


def fill_values_A(array):
    mean = array.mean()
    array_A = array.fillna({"Age": mean["Age"],
                          "GPA": mean["GPA"],
                          "Days_missed": mean["Days_missed"]})
    array_A.to_csv("mock_student_data_A.csv")


def fill_values_B(array):
    mean_yes = array.loc[array["Graduated"] == "Yes"].mean()
    mean_no = array.loc[array["Graduated"] == "No"].mean()

    '''
    array.loc[(array["Graduated"] == "Yes") & (p.isnull(array["Age"]))] = mean_yes["Age"]
    array.loc[(array["Graduated"] == "No") & (p.isnull(array["Age"]))] = mean_no["Age"]
    '''
    array_yes = array.loc[array["Graduated"] == "Yes"].fillna({"Age": mean_yes["Age"],
                          "GPA": mean_yes["GPA"],
                          "Days_missed": mean_yes["Days_missed"]})

    array_no = array.loc[array["Graduated"] == "No"].fillna({"Age": mean_no["Age"],
                          "GPA": mean_no["GPA"],
                          "Days_missed": mean_no["Days_missed"]})

    array_B = array_yes.join(array_no, on = "index")

    '''
    array.loc[array["Graduated"] == "Yes"].fillna({"Age": mean_yes["Age"],
                          "GPA": mean_yes["GPA"],
                          "Days_missed": mean_yes["Days_missed"]})

    array_B = array.fillna({"Age": mean["Age"],
                          "GPA": mean["GPA"],
                          "Days_missed": mean["Days_missed"]})
    array_B.to_csv("mock_student_data_B.csv")
    '''

if __name__ == '__main__':
    filename = "mock_student_data.csv"
    array = read_data(filename)
    
    summarize_data(array)
    graph_data(array)
    
    get_genders(array)