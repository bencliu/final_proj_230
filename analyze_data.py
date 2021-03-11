import numpy as np
import pickle
import matplotlib.pyplot as plot
import numpy as np
import pprint

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


    """# read in county_areas
    with open('data/county_areas.p', 'rb') as fp:
        countyarea = pickle.load(fp)
    with open('data/aws_id_areas.p', 'rb') as fp:
        awsarea = pickle.load(fp)
    dataaws = []
    for key, val in awsarea.items():
        dataaws.append(val)
    plot.hist(dataaws, density=True, bins=1000)
    data = []
    for key, val in countyarea.items():
        data.append(val)
    plot.hist(data, density=True, bins=1000)
    plot.show()"""
    #pprint.pprint(countyarea)
