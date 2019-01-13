import scipy.io as sio
import numpy as np

def load_data():
    data_awgn = sio.loadmat("datasets/mnist-with-awgn.mat")
    data_rc_awgn = sio.loadmat("datasets/mnist-with-reduced-contrast-and-awgn.mat")
    data_motion_blur = sio.loadmat("datasets/mnist-with-motion-blur.mat")

    train_x = np.concatenate((data_awgn['train_x'], data_rc_awgn['train_x'], data_motion_blur['train_x']), axis=0)
    train_y = np.concatenate((data_awgn['train_y'], data_rc_awgn['train_y'], data_motion_blur['train_y']), axis=0)
    test_x = np.concatenate((data_awgn['test_x'], data_rc_awgn['test_x'], data_motion_blur['test_x']), axis=0)
    test_y = np.concatenate((data_awgn['test_y'], data_rc_awgn['test_y'], data_motion_blur['test_y']), axis=0)
    return ((train_x, train_y), (test_x, test_y))
