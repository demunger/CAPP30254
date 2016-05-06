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
    ## Return a list of column headers containing missing data
    missing = df.isnull().sum()
    return [missing.index[x] for x in list(filter(lambda x: missing[x] > 0, range(len(missing))))]


def get_imputed_value(df, column, method = "mean"):
    ## Compute the mean, median, or mode of a select column by header
    if method == "median":
        return df[column].median();
    elif method == "mode":
        return df[column].mode()[0];
    else:
        return df[column].mean();


def impute_values(df):
    ## Impute missing data; return a dictionary of column and value pairs
    value_dict = {}
    missing_columns = get_missing_columns(df)

    for column in missing_columns:
        value_dict[column] = get_imputed_value(df, column)
    df.fillna(value_dict, inplace = True)

    return value_dict


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


def log_values(df, columns):
    ## Take the log value in field column
    df[columns] = df[columns].apply(lambda x: np.log(x + 1))

'''
def logistic_regression(y, X):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def k_nearest_neighbor(y, X):
    model = KNeighborsClassifier()
    model.fit(X, y)
    return model


def linear_svm(y, X):
    model = LinearSVC()
    model.fit(X, y)
    return model
'''

def test_accuracy(model, y, X):
    return model.score(X, y)


def predict_model(model, df):
    return model.predict(df)


if __name__ == '__main__':
    num_args = len(sys.argv)

    if num_args == 1:
        training_filename = "cs-training-short.csv"
        scoring_filename = "cs-test.csv"
        dependent_column = 0
    elif num_args == 4:
        training_filename = sys.argv[1]
        scoring_filename = sys.argv[2]
        dependent_column = sys.argv[3]
    else:
        print("Usage: {} <training_data.csv> <scoring_data.csv> <dependent variable column #>".format(sys.argv[0]))
        sys.exit(0)
    
    ## Read data
    training_data = read_data(training_filename)
    '''
    ## Explore data
    summarize_data(training_data)
    graph_data(training_data)
    '''
    ## Pre-process data
    value_dict = impute_values(training_data)

    ## Generate features
    discretize_ratios(training_data, ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"])
    log_values(training_data, "MonthlyIncome")
    standardize_values(training_data, ["age", "NumberOfTime30-59DaysPastDueNotWorse", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"])

    ## Build classifier
    models_framework = [("Logit", LogisticRegression()), ("K-NN", KNeighborsClassifier()), ("SVM", LinearSVC())]
    
    ## Separate dependent, independent variable
    y = training_data[training_data.columns[dependent_column]]
    X = training_data.drop(training_data.columns[dependent_column], axis = 1)
    built_models = []
    for clf in models_framework
    model[1].fit(y, X) for model in models_framework]


    ## Evaluate classifier
    accuracy = [test_accuracy(model, y, X) for model in built_models]
    print("Accuracy on training data for:")
    for i, percent in enumerate(accuracy):
        print("\t{}: \t{:.1%}".format(models_framework[i][0], percent))

    ## Score new data
    scoring_data = read_data(scoring_filename)
    ## Pre-process scoring data
    scoring_data = scoring_data.drop(scoring_data.columns[dependent_column], axis = 1)
    scoring_data.fillna(value_dict, inplace = True)

    ## Select highest performing model
    model = built_models[accuracy.index(max(accuracy))]
    prediction = predict_model(model, scoring_data) 
    np.savetxt("predicted_values.csv", prediction)
