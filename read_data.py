import numpy as np

def load_data(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Creates one hot representation of labels
def create_onehot(labels):
    labels_onehot = np.zeros((len(labels),10))
    counter = 0
    for i in labels:
        labels_onehot[counter][i] = 1
        counter += 1
    return labels_onehot


# Reads the cifar 10 data batches and returns meta data, training and test data and labels
def get_data():
    # datafiles
    traindata_file1 = "cifar-10-batches-py/data_batch_1"
    traindata_file2 = "cifar-10-batches-py/data_batch_2"
    traindata_file3 = "cifar-10-batches-py/data_batch_3"
    traindata_file4 = "cifar-10-batches-py/data_batch_4"
    traindata_file5 = "cifar-10-batches-py/data_batch_5"
    testdata_file = "cifar-10-batches-py/test_batch"
    meta_file = "cifar-10-batches-py/batches.meta"

    # load the data to a python dictionary
    train_batch1 = load_data(traindata_file1)
    train_batch2 = load_data(traindata_file2)
    train_batch3 = load_data(traindata_file3)
    train_batch4 = load_data(traindata_file4)
    train_batch5 = load_data(traindata_file5)
    test_batch = load_data(testdata_file)
    meta_data = load_data(meta_file)

    # extract data and labels from the dictionary
    train_data1 = train_batch1["data".encode()]
    train_labels1 = train_batch1["labels".encode()]
    train_data2 = train_batch2["data".encode()]
    train_labels2 = train_batch2["labels".encode()]
    train_data3 = train_batch3["data".encode()]
    train_labels3 = train_batch3["labels".encode()]
    train_data4 = train_batch4["data".encode()]
    train_labels4 = train_batch4["labels".encode()]
    train_data5 = train_batch5["data".encode()]
    train_labels5 = train_batch5["labels".encode()]
    test_data = test_batch["data".encode()]
    test_labels = test_batch["labels".encode()]

    # merge the train data and labels together
    train_data = np.concatenate((train_data1, train_data2, train_data3, train_data4, train_data5), axis=0)
    train_labels = np.concatenate((train_labels1, train_labels2, train_labels3, train_labels4, train_labels5), axis=0)

    # create a one hot representation of the labeled data
    train_labels = create_onehot(train_labels)
    test_labels = create_onehot(test_labels)

    return train_data, train_labels, test_data, test_labels, meta_data