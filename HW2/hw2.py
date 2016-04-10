import pandas as pd
import sys

def read_data(filename):
    return p.read_csv(filename)

if __name__ == '__main__':
    num_args = len(sys.argv)

    if num_args != 3:
        print("Usage: {} <training_data.csv> <testing_data.csv>".format(sys.argv[0]))
        sys.exit(0)

    training_filename = sys.argv[1]
    testing_filename = sys.argv[2]
    
    ## Read data
    training_data = read_data(training_filename)
    testing_data = read_data(testing_filename)

    ## Explore data
    training_data.describe()

    ## Pre-process data

    ## Generate features

    ## Build classifier

    ## Evaluate classifier