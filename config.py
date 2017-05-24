# adjust hyper-parameter
import numpy as np

class base_config(object):
    def __init__(self, train_data, test_data):
        self.train_range = np.array(range(len(train_data)))
        self.test_range = np.array(range(len(test_data)))
        self.batch_size = 64
        self.lr = 0.0001
        self.num_classes = 5


        self.image_size = 224
        self.channels = 3

        self.max_epoch = 10
