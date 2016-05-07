import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


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


def split_data(df, dependent_column = 0, testing_split = .2):
    ## Separate dependent, independent variable
    y = training_data[training_data.columns[dependent_column]]
    X = training_data.drop(training_data.columns[dependent_column], axis = 1)
    ## Create 80/20 train/test split
    return train_test_split(X, y, test_size = 0.2, random_state = 0)


def magic_loop(models_framework, grid_framework, built_models, X_train, X_test, y_train, y_test):
    for index, clf in models_framework.items():
        parameters = grid_framework[index]
        
        ## Measure time to fit classifier based on parameters
        start_time = time.time()

        grid = GridSearchCV(estimator = clf, param_grid = parameters, verbose = 10)
        grid.fit(X_train, y_train)

        ## Store model attributes
        built_models[index]["time"] = time.time() - start_time
        built_models[index]["best_model"] = grid.best_estimator_ 
        built_models[index]["best_params"] = grid.best_params_
        built_models[index]["best_score"] = grid.best_score_
        
        y_predict = built_models[index]["best_model"].predict(X_test)
        ## Store model metrics
        built_models[index]["accuracy"] = accuracy_score(y_test, y_predict)
        built_models[index]["precision"] = precision_score(y_test, y_predict)
        built_models[index]["recall"] = recall_score(y_test, y_predict)
        built_models[index]["F1"] = f1_score(y_test, y_predict)
        built_models[index]["AUC"] = roc_auc_score(y_test, y_predict)

        if hasattr(clf, "predict_proba"):
            y_prob = built_models[index]["best_model"].predict_proba(X_test)[:,1]
        else:
            y_prob = built_models[index]["best_model"].decision_function(X_test)

        plot_precision_recall(y_test, y_prob, index)


def plot_precision_recall(y_test, y_prob, index):
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)

    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]

    plot_precision_and_recall(index, y_test, y_prob, recall_curve, precision_curve, pr_thresholds)
    plot_precision_v_recall(index, y_test, y_prob, recall_curve, precision_curve)


def plot_precision_and_recall(index, y_test, y_prob, recall_curve, precision_curve, pr_thresholds):
    num = len(y_prob)
    pct_above_per_thresh = []
    for value in pr_thresholds:
        num_above_thresh = len(y_prob[y_prob >= value])
        pct_above_thresh = num_above_thresh / float(num)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, "blue")
    ax1.set_xlabel("Percent of Population")
    ax1.set_ylabel("Precision", color = "blue")
    
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, "red")
    ax2.set_ylabel("Recall", color = "red")
    plt.title(index)
    plt.show()


def plot_precision_v_recall(index, y_test, y_prob, recall_curve, precision_curve):
    avg_precision = average_precision_score(y_test, y_prob)
    
    plt.clf()
    plt.plot(recall_curve, precision_curve, label = "area = {:0.2}".format(avg_precision))
    plt.legend(loc = "lower right")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title(index)
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.show()


if __name__ == "__main__":
    num_args = len(sys.argv)

    if num_args == 1:
        training_filename = "cs-training.csv"
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

    ## Explore data
    summarize_data(training_data)
    graph_data(training_data)

    ## Remove outliers
    training_data = training_data[training_data["age"] > 0]

    ## Pre-process data
    value_dict = impute_values(training_data)

    ## Generate features
    discretize_ratios(training_data, ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"])
    log_values(training_data, "MonthlyIncome")
    standardize_values(training_data, ["age", "NumberOfTime30-59DaysPastDueNotWorse", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"])

    ## Build classifier
    models_framework = {"Logit": LogisticRegression(),
                        "K-NN": KNeighborsClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "SVM": LinearSVC(),
                        "Random Forest": RandomForestClassifier(),
                        "Boosting": GradientBoostingClassifier(),
                        "K-NN Bagging": BaggingClassifier(KNeighborsClassifier())}

    grid_framework = {"Logit": {"penalty": ["l1", "l2"], 
                               "C": [0.01, 0.1, 1, 10]},
                      "K-NN": {"n_neighbors": [1, 5, 10],
                               "weights": ["uniform", "distance"],
                               "algorithm": ["auto", "ball_tree", "kd_tree"]},
                      "Decision Tree": {"criterion": ["gini", "entropy"], 
                               "max_depth": [1, 5], 
                               "max_features": ["sqrt", "log2"],
                               "min_samples_split": [2, 10]},
                      "SVM": {"C": [0.01, 0.1, 1]},
                      "Random Forest": {"n_estimators": [1, 10, 50], 
                               "max_depth": [1, 5], 
                               "max_features": ["sqrt", "log2"],
                               "min_samples_split": [2, 10]},
                      "Boosting": {"n_estimators": [1, 10], 
                               "learning_rate": [0.001, 0.01, 0.1],
                               "subsample": [0.1, 0.5, 1.0], 
                               "max_depth": [1, 5]},
                      "K-NN Bagging": {"max_samples": [0.1, 0.5, 1.0],
                               "max_features": [0.1, 0.5, 1.0]}}

    model_attributes = ["best_params", "best_score", "best_model", "time"]
    model_metrics = ["accuracy", "precision", "recall", "F1", "AUC"]
    built_models = {model: dict.fromkeys(model_attributes + model_metrics) for model in list(models_framework.keys())}

    magic_loop(models_framework, grid_framework, built_models, *split_data(training_data))

    ## Score new data
    scoring_data = read_data(scoring_filename)
    ## Pre-process scoring data
    scoring_data = scoring_data.drop(scoring_data.columns[dependent_column], axis = 1)
    scoring_data.fillna(value_dict, inplace = True)
    ## Find max AUC
    max_auc = 0
    for model in built_models.keys():
        if(max_auc < built_models[model]["AUC"]):
            max_auc = built_models[model]["AUC"]
            new_model = built_models[model]["best_model"]
    prediction = predict_model(new_model, scoring_data) 
    np.savetxt("predicted_values.csv", prediction)