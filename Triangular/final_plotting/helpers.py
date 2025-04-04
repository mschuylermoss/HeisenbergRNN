import pickle

def save_dict(dict_,filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict_, f)


def load_dict(filename):
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict
