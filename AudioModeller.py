import os
import pickle
from datetime import time
from scipy.io.wavfile import read
import numpy as np
from sklearn.mixture import GaussianMixture

from FeatureExtracter import extract_features


def train_model():
    source = "C:\\Users\\aimanov\\Speaker-Identification-Using-Machine-Learning\\training_set\\"
    dest = "C:\\Users\\aimanov\\Speaker-Identification-Using-Machine-Learning\\trained_models\\"
    train_file = "C:\\Users\\aimanov\\Speaker-Identification-Using-Machine-Learning\\training_set_addition.txt"
    file_paths = open(train_file, 'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        print(path)

        sr, audio = read(source + path)
        print(sr)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
            features = np.asarray(())
            count = 0
        count = count + 1


def test_model():
    source = "C:\\Users\\aimanov\\Speaker-Identification-Using-Machine-Learning\\testing_set\\"
    modelpath = "C:\\Users\\aimanov\\Speaker-Identification-Using-Machine-Learning\\trained_models\\"
    test_file = "C:\\Users\\aimanov\\Speaker-Identification-Using-Machine-Learning\\testing_set_addition.txt"
    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]

    # Read the test directory and get the list of test audio files
    for path in file_paths:

        path = path.strip()
        print(path)
        sr, audio = os.read(source + path)
        vector = extract_features(audio, sr)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        time.sleep(1.0)
    # choice=int(input("\n1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n"))
