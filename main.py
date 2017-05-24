import tensorflow as tf
import numpy as np
import sys

from config import base_config as config
from models import convNet
from flower_data import load_data

argvs = sys.argv

def run_task(data_dir, task_name = ''):

    train_data, test_data = load_data(data_dir)
    conv_config = config(train_data, test_data)

    with tf.Session() as sess:
        model = convNet(conv_config, sess, train_data, test_data)
        model.build_model()
        if len(argvs) < 2:
            model.run(task_name)
        else:
            model.demo(argvs[1])

def main(_):
    
    run_task('flowers/', 'flower_recognition')

if __name__ == '__main__':
    tf.app.run()



