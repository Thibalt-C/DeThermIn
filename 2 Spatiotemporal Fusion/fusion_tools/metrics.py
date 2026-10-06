import numpy as np

def rmse(x,y):
    return np.sqrt(np.nansum((x-y)**2)/len(x.flatten()))

def normalised_rmse(x, y, eps=270):
    # epsilon to ensure always positive values
        return np.sqrt(np.nansum(((x-y)/(y+eps))**2)/len(x.flatten()))