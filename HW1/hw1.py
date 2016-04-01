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


def get_genders(array):
    '''
    for index, row in array.iterrows():
        if p.isnull(row["Gender"]):
            request = requests.get("http://api.genderize.io?name=" + row["First_name"])
            row["Gender"] = json.loads(request.text)["gender"].title()
    return array
    '''
    for i in array.ix[p.isnull(array["Gender"])].index:
        request = requests.get("http://api.genderize.io?name=" + array["First_name"][i])
        array["Gender"][i] = json.loads(request.text)["gender"].title()

    '''
    name_list = [array["First_name"][index] for index, row in array.ix[p.isnull(array["Gender"])].iterrows()]
    "&".join(["name[{}]={}".format(i, x) for i, x in enumerate(name_list)])
    request = requests.get("http://api.genderize.io?" +
    '''


def fill_values_A(array):
    mean = array.mean()
    array_A = array.fillna({"Age": mean["Age"],
                          "GPA": mean["GPA"],
                          "Days_missed": mean["Days_missed"]})
    array_A.to_csv("mock_student_data_A.csv")


def fill_values_B(array):
    mean_yes = array.ix[array["Graduated"] == "Yes"].mean()
    mean_no = array.ix[array["Graduated"] == "No"].mean()

    array.ix[(array["Graduated"] == "Yes") & (p.isnull(array["Age"])), "Age"] = mean_yes["Age"]
    array.ix[(array["Graduated"] == "No") & (p.isnull(array["Age"])), "Age"] = mean_no["Age"]

    array.ix[(array["Graduated"] == "Yes") & (p.isnull(array["GPA"])), "GPA"] = mean_yes["GPA"]
    array.ix[(array["Graduated"] == "No") & (p.isnull(array["GPA"])), "GPA"] = mean_no["GPA"]

    array.ix[(array["Graduated"] == "Yes") & (p.isnull(array["Days_missed"])), "Days_missed"] = mean_yes["Days_missed"]
    array.ix[(array["Graduated"] == "No") & (p.isnull(array["Days_missed"])), "Days_missed"] = mean_no["Days_missed"]
    

    '''    
    array_yes = array.loc[array["Graduated"] == "Yes"].fillna({"Age": mean_yes["Age"],
                          "GPA": mean_yes["GPA"],
                          "Days_missed": mean_yes["Days_missed"]})

    array_no = array.loc[array["Graduated"] == "No"].fillna({"Age": mean_no["Age"],
                          "GPA": mean_no["GPA"],
                          "Days_missed": mean_no["Days_missed"]})

    array_B = array_yes.merge(array_no, how = "outer")
    '''

    array.to_csv("mock_student_data_B.csv")
    

if __name__ == '__main__':
    filename = "mock_student_data.csv"
    array = read_data(filename)
    
    summarize_data(array)
    graph_data(array)
    
    get_genders(array)