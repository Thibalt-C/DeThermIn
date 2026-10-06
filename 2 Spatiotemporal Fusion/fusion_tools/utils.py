import pickle

def import_dataset(filename):

    dataset = pickle.load(open(filename, "rb"))

    return dataset