import glob
import random
import cPickle
from scipy import ndimage, misc
import os

label_dict = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4
}

WIDTH = 224
HEIGHT = 224

def save_pkl(obj, path):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)

def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

def image_label(file_name):
    for name in label_dict.keys():
        if name in file_name:
            return label_dict[name]
    
    print(file_name)
    raise "unknown label %s" % file_name

def load_data(data_dir = 'flowers/'):
    train_path = data_dir + 'train.pkl'
    test_path = data_dir + 'test.pkl'

    train_data = None
    test_data = None
    if os.path.exists(train_path):
        train_data = load_pkl(train_path)
        test_data = load_pkl(test_path)
    else:
        train_data, test_data = data2pickle(data_dir)

    return train_data, test_data

def data2pickle(data_dir = 'flowers/'):

    train_files = glob.glob('%sraw-data/train/*/*.jpg' % data_dir)
    test_files = glob.glob('%sraw-data/validation/*/*.jpg' % data_dir)

    random.shuffle(train_files)

    train_data = []
    for f in train_files:
        im = ndimage.imread(f)
        k = misc.imresize(im, (WIDTH, HEIGHT))
        v = image_label(f)
        train_data.append((k, v))

    test_data = []
    for f in test_files:
        im = ndimage.imread(f)
        k = misc.imresize(im, (WIDTH, HEIGHT))
        v = image_label(f)
        test_data.append((k, v))
    
    train_path = data_dir + 'train.pkl'
    test_path = data_dir + 'test.pkl'

    save_pkl(train_data, train_path)
    save_pkl(test_data, test_path)

    return train_data, test_data
