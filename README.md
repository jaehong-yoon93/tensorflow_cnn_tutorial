# tensorflow_cnn_tutorial

This repository is for practicing a tensorflow only



## description

Answer.py:
rename answer.py to Convolution.py if you want to apply answer.py

or

fill in Convolution.py like as answer.py(see models/answer.py). funtion run(), build_model(), demo() are little different from original


## usage

To download:

    $ ./download_and_preprocess_flowers.sh flowers

train:

    $ python main.py

demo:

    $ python main.py test/daisy.jpg

For demo, give the 'file_path' as a third argument. demo should be excuted after train.

## Feedforward to CNN

Run feedforward network using mnist,

    $ python mnist_tutorial.py

Run CNN using mnist, 

    $ python mnist_cnn_ver.py

This network has two convolutional layers and two fully-connected layers.
