# AlphaOthello
My first working version of alphazero on Othello, trained on my laptop for three month and achieved 6-dan level on Othello Quest app

# Background
I built this at the end of my freshmen year. I didn't have the habit of publishing my code, but recently I lost a lot of my favorite code
when I removed my broken linux system before I realize that the code were not successfully copied to my usb drive. It's been a long time since the last time
I worked on this code because I got a satisfying result and started working on other things.
A lot of modification to accelerate the code so that I could train a decent model, so I am going to write about how to use the code based on my memory.

# Usage
Install a correct version of cuda and tensorrt (I kind of forgot which version)
requirements: opencv-python, tensorflow, keras, Cython (a version that allows "cpdef", or I will need to maintain the code), Pygame (I will need to check if there are any other things)

If in windows, execute `build.bat` so that it converts `mcts.pyx` and `Othello.pyx` to C programs (we need a working Cython version for this). We also need to prepare
a `data` folder in the root directory to store selfplay data. 

# Generate selfplay data
run `init.py` to first create a random neural network

run `selfplay.py`, it will store self-play games in the `data` folder, but remember to change the `example_count` variable in this file just to store selfplay data into
different files

run `train-v2.py` to load from `data` folder to train a model, but adjust the `min_c` and `max_c` to choose the portion of data to load. This only trains the keras model

`keras_to_tf.py` converts the keras model to tensorflow model, and `tf_to_trt.py` converts tensorflow model to tensorRT model. This needs to be repeated everytime before training (it is annoying, sorry that I was not very good at coding at that time). My training code is only used for training keras models, but tensorRT models are much faster in generating self play games.

`play-v2.py` is a graphical user interface, running this script will initiate a match against the trained tensorRT model.

# Results
Posting all the training data gathered in the releases folder.

