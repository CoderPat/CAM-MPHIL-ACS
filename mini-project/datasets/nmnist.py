import scipy.io as sio

def load_data():
    data = sio.loadmat("datasets/mnist-with-awgn.mat")
    return (data['train_x'], data['train_y']), (data['test_x'], data['test_y'])
