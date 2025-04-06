notes.md

## Run on Linux (Ubuntu on cloud)
Choosing
* python `3.14.0a6t` or `3.13.2t` or `3.13.1`? oops: It says: Python 3.8–3.11. O `3.12.9` worked.
* TF2 `v2.19.0`
* I am migrating From TF 1.15.0 to 2.19.0.


https://www.tensorflow.org/install
I=Also a dockerbsed solution exists, just in case it was needed:
```bash
 docker pull tensorflow/tensorflow:latest  # Download latest stable image
 docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server
```
Provisionaing (one-off per dev machine)
* `pyenv install 3.12.9`
* `pyenv local 3.12.9`
Installation:
* `python --version`
* `pip install --upgrade pip`
* `pip install tensorflow `    `#`installed 2.19.0
* `pip install tensorflow==2.19.0`
* `pip install scipy imageio matplotlib scikit-image`
Run:
* `python glyphnet/glyphnet1.py`
* Note: No `source` command needed. CWD is repo root, e.g. `/dataneura/glyphnet/glyphnet`

# Deprecated notes

## Run on MacOS


MacOS:
Installation on MacOS: (First time only)
    * virtualenv --version # If error, install virsualenv . see https://www.tensorflow.org/install/pip
    * cd glyphnet
    * virtualenv -v --python=python3  ./tensorf1
    * source ./tensorf1/bin/activate
    * pip install tensorflow==1.15.0
    * pip install scipy
    * pip install imageio
    * pip install  matplotlib
    * pip install scikit-image

    Unsure: cython PyHamcrest

Run on MacOS
    * cd glyphnet
    * source ./tensorf1/bin/activate
    * python glyphnet1.py


## Run on Linux
MNIST only (unused)
```bash
PYTHONPATH=. python ./bin/mnistgan.py
```

## Run on Windows (Anaconda)


## Tensorboard
 1. Add two lines to code.
    ```python
        graph_writer = tf.summary.FileWriter("./graph/", sess.graph)
        graph_writer.close()
    ```
 2. Run in commandline:   `tensorboard --logdir="./graph"`
 3. Browse:  `http://localhost:6006/`
See: https://www.tensorflow.org/guide/graphs

## Misc notes
Based on:
https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/406_GAN.py

### delete:
tf.layers.conv2d
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
[batch_size, image_height, image_width, channels]
print(tf.__file__)


Read misc:
CNN:
https://www.tensorflow.org/tutorials/estimators/cnn
GAN + conv2d:
https://datascience.stackexchange.com/questions/30810/gan-with-conv2d-using-tensorflow-shape-error
morvanzhou:
https://morvanzhou.github.io/tutorials/
https://www.youtube.com/user/MorvanZhou




## Run on Linux
linux preparation (ubuntu 16)

The tensorflow 2 does not work yet.
virtualenv -v --python=python3  ./tensorf2
pip install tensorflow
(failed)

Use Tensorflow 1 instead:
# virtualenv -v --python=python3  ./tensorf1
virtualenv -v --python=python3  ~/.virtualenvs/tensorf1
source ~/.virtualenvs/tensorf1/bin/activate

pip install tensorflow==1.15.0


pip install scipy
pip install imageio
pip install matplotlib
pip install scikit-image

sudo apt-get install tcl-dev tk-dev python-tk python3-tk

every time:
source ~/.virtualenvs/tensorf1/bin/activate
