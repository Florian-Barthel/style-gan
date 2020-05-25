import pickle
import os

import config


def load_pkl(file):
    with open(file, 'rb') as file:
        return pickle.load(file, encoding='latin1')


def save_pkl(obj, description, iteration):
    models_folder = config.result_folder + '/models/'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    with open(models_folder + '/{}_model_at_iteration{:04d}.pickle'.format(description, iteration), 'wb') as file:
        pickle.dump(obj, file)
