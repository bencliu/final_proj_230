import numpy as np
import pickle
import matplotlib.pyplot as plot
import numpy as np
import pprint
import pandas as pd

"""
Function: Plot histogram of data
@Params: File path of training data label dictionary
@Return: none
"""
def plot_training_data_histogram(file: str):

    # Read in label pickle file
    with open(file, 'rb') as fp:
        labels = pickle.load(fp)

    # assemble list of yield data
    data = []
    for key, val in labels.items():
        data.append(val)

    # plot histogram
    plot.hist(data, density=True, bins=500)
    plot.xlabel('Scaled soybean yield (Normalized Bushel/Acre per County Area)')
    plot.ylabel('Occurence')
    plot.title('Histogram of Training Labels')
    plot.xlim(left=-5, right=10)
    plot.grid()
    plot.show()

"""
Function: Process Training Log
"""
def process_training_log(filepath: str, name_of_model: str):

    # read in raw file
    df = pd.read_csv(filepath)

    # plot train/val error as a function of epoch
    df.plot(x = "epoch", y = ["val_mean_squared_error", "mean_squared_error"])
    plot.ylabel("MSE")
    plot.title(name_of_model + " train/val MSE")
    plot.xlim(left = 0, right = len(df["epoch"])+1 )
    plot.grid()
    plot.show()

"""
Function: pre-processes the training data by removing the mean and scaling by the variance
@Params: file path of dictionary of lavel data
"""
def preprocess_data(file: str):

    # read in label pickle file
    with open(file, 'rb') as fp:
        label_dict = pickle.load(fp)

    # extract values and perform normalization
    labels = np.array(list(label_dict.values()))
    mean = np.mean(labels)
    labels = labels - mean
    variance = np.var(labels)
    labels = labels/variance

    # write back to dictionary
    i = 0
    for key, val in label_dict.items():
        label_dict[key] = labels[i]
        i += 1

    # write back to pickle file
    with open('data/processed_label_dict.p', 'wb') as fp:
        pickle.dump(label_dict, fp)



if __name__ == "__main__":

    # preprocess_data('data/master_label_dict_v3.p')
    plot_training_data_histogram('data/processed_label_dict.p')
    # process_training_log("concat_model_results/concat_run_1.log", "Concatenated Arms Model")


