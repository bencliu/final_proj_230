import numpy as np
import pickle
import matplotlib.pyplot as plot
import numpy as np

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
    plot.hist(data, density=True, bins=100)
    plot.xlabel('Soybean Yield per image (Bushel/Acre)')
    plot.ylabel('Occurence')
    plot.title('Histogram of Training Data')
    plot.grid()
    plot.show()
    print(sum(data)/len(data))

if __name__ == "__main__":

    plot_training_data_histogram('json_store/labels_v2/master_label_dict_vUpdate.pkl')
