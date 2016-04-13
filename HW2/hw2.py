import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

def read_data(filename):
    df = pd.read_csv(filename)
    ## Ommit ID column
    return df.drop(df.columns[[0]], axis = 1)


def summarize_data(df):
    ## General description
    print(np.round(df.describe(percentiles = [.5]), 2).to_string(justify = "left"))

    ## Summary statistics
    mean = df.mean().to_string(float_format = "{:.2f}".format)
    print("Mean:\n" + mean + "\n")

    median = df.median().to_string(float_format = "{:.2f}".format)
    print("Median:\n" + median + "\n")

    std_deviation = df.std().to_string(float_format = "{:.2f}".format)
    print("Standard Deviation:\n" + std_deviation + "\n")

    mode = df.mode().to_string(index = False)
    print("Mode:\n" + mode + "\n")

    missing_values = df.isnull().sum().to_string()
    print("Missing Values:\n" + missing_values + "\n")


def graph_data(df):
    ## Graph classifier
    df.groupby(df.columns[0]).size().plot(kind = "bar", width = 1, rot = 0)
    plt.show()

    ## Graph column distributions
    for name in df.columns[1:]:
        df.groupby(name).size().plot()
        plt.show()


def get_missing_columns(df):
    missing = df.isnull().sum() 
    return [missing.index[x] for x in list(filter(lambda x: missing[x] > 0, range(len(missing))))]


def condition_means_imputation(df, missing_columns):
    positive_df = df.ix[df[df.columns[0]] == 0]
    negative_df = df.ix[df[df.columns[0]] == 1]

    mean_positive = positive_df.mean()
    mean_negative = negative_df.mean()

    positive_df = fill_values(positive_df, missing_columns, mean_positive)
    negative_df = fill_values(negative_df, missing_columns, mean_negative)
    return pd.concat([positive_df, negative_df])


def fill_values(df, missing_columns, values):
    value_dict = {}
    for value in missing_columns:
        value_dict[value] = values[value]

    return df.fillna(value_dict)    


def discretize_ratios(df, columns):
    ## Create [0,1] value bounds for a for percent/ratio field column
    for column in columns:
        df.ix[df[column] > 1] = 1


def binary_dummy(df, column):
    ## Create a dummy variable for each n-1 unique values in field column
    for value in df[column].unique()[:-1]:
        df[column + "=" + str(value)] = df[column] == value


def standardize_values(df, columns):
    ## Mean standardize values in field column
    df[columns] = df[columns].apply(lambda x: (x - x.mean()) / x.std())


def logistic_regression(df):
    model = LogisticRegression()
    model.fit(df[df.columns[1:]], df[df.columns[0]])
    return model


def k_nearest_neighbor(df):
    model = KNeighborsClassifier()
    model.fit(df[df.columns[1:]], df[df.columns[0]])
    return model


def linear_svc(df):
    model = LinearSVC()
    model.fit(df[df.columns[1:]], df[df.columns[0]])
    return model


def test_accuracy(model, df):
    return model.score(df[df.columns[1:]], df[df.columns[0]])


def fit_model(model, df):
    return model.predict(df[df.columns[1:]])


if __name__ == '__main__':
    num_args = len(sys.argv)

    if num_args == 1:
        training_filename = "cs-training.csv"
        scoring_filename = "cs-test.csv"
    elif num_args == 3:
        training_filename = sys.argv[1]
        scoring_filename = sys.argv[2]
    else:
        print("Usage: {} <training_data.csv> <scoring_data.csv>".format(sys.argv[0]))
        sys.exit(0)
    
    ## Read data
    training_data = read_data(training_filename)

    ## Explore data
    summarize_data(training_data)
    graph_data(training_data)

    ## Pre-process data
    missing_columns = get_missing_columns(training_data)
    training_data = condition_means_imputation(training_data, missing_columns)

    ## Generate features
    discretize_ratios(training_data, ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"])
    binary_dummy(training_data, "age")
    standardize_values(training_data, ["NumberOfTime30-59DaysPastDueNotWorse", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"])

    ## Build classifier
    models_framework = [("Logit", logistic_regression), ("K-NN", k_nearest_neighbor), ("SVC", linear_svc)]
    built_models = [model[1](training_data) for model in models_framework]

    ## Evaluate classifier
    accuracy = [test_accuracy(model, training_data) for model in built_models]
    print("Accuracy on training data for:")
    for i, percent in enumerate(accuracy):
        print("\t{}: \t{:.1%}".format(models_framework[i][0], percent))

    ## Score new data
    model = built_models[accuracy.index(max(accuracy))]
    scoring_data = read_data(scoring_filename)
    ## Remove missing data and classifier column
    scoring_data = scoring_data.drop(scoring_data.columns[[0]], axis = 1).dropna()
    fit_model(model, scoring_data).to_csv("predicted_values.csv", index = False)