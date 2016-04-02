import pandas as p
import requests
import json


def read_data(filename):
    return p.read_csv(filename)


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
    for i in array.ix[p.isnull(array["Gender"])].index:
        #request = requests.get("http://api.genderize.io?name=" + array["First_name"][i])
        array.loc[i, "Gender"] = "A" #json.loads(request.text)["gender"].title()


def fill_values_A(array):
    mean = array.mean().round(1)
    array_A = fill_values(array, mean)
    array_A.to_csv("mock_student_data_A.csv", index = False)


def fill_values_B(array):
    grad = array.ix[array["Graduated"] == "Yes"]
    non_grad = array.ix[array["Graduated"] == "No"]

    mean_grad = grad.mean().round(1)
    mean_non_grad = non_grad.mean().round(1)

    array_B = p.concat([fill_values(grad, mean_grad), fill_values(non_grad, mean_non_grad)]).sort_index(by = "ID")
    
    array_B.to_csv("mock_student_data_B.csv", index = False)
    

def fill_values(array, values):
    return array.fillna({"Age": values["Age"],
                         "GPA": values["GPA"],
                         "Days_missed": values["Days_missed"]})


if __name__ == '__main__':
    '''
    ## Load student record file
    filename = "mock_student_data.csv"
    array = read_data(filename)
    
    ## Generate summary statistics
    sum_array = array.ix[:, 1:] # Ommit ID column
    summarize_data(sum_array)
    graph_data(sum_array)
    
    ## Infer missing genders
    get_genders(array)

    ## Infer missing data using mean, class-conditional mean values
    fill_values_A(array)
    fill_values_B(array)
    '''