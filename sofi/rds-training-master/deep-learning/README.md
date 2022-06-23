## Deep Learning Tutorial Gateway
---

### Overview
---

What is deep learning:

It is just a model architecture is that very `flexible` that is very good at dealing with challenges that traditional modeling techniques cannot address.
* sequence modeling
    * natual language processing
    * voice recognition
    * signal processing
* computer vision
    * image classification
    * image generation
    * image segmentation
* generative tasks
    * data generalization
    * image/text/audio generalization
* reinforcement learning
    * training a robot
* combination of the tasks above
    * like visual question answering
* ...

### Learning Materials
---
Readings
* [Overview of Deep Learning](https://lilianweng.github.io/lil-log/2017/06/21/an-overview-of-deep-learning.html)
* [Learn Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning)
* [more technical slides on CNN and RNN](http://www.cs.cmu.edu/~mgormley/courses/10601/slides/lecture13-NN2.pdf)

Videos
* MLP: [video](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown)
    * This channel has a collection on Neural Nets
* CNN: [video](https://www.youtube.com/watch?v=YRhxdVk_sIs&ab_channel=deeplizard)
* RNN: [video](https://www.youtube.com/watch?v=LHXXI4-IEns&ab_channel=TheA.I.Hacker-MichaelPhi)
    * [reading](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

Frameworks
* [Tensorflow/Keras](https://www.tensorflow.org/tutorials/keras/classification)
* [PyTorch](https://pytorch.org/tutorials/)

State of the art model types:
* [Papers with Code](https://paperswithcode.com/)
* Arxiv
* Youtube - talks and stuff
* Google scholar

### Pro and Cons
--- 

But deep learning is not a panaca - no free lunch theorem! Here is a list of reasons of when and when not to use deep learning

Pros
* It is an `universal function approximator`. Read more [here](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6)
* It is essentially a feature extractor.
    * Notice what the aforemented list of tasks have in common? It is difficult to manually build features for all of the tasks.
    * e.g. how to define a feature that captures a human face?
    * e.g. how to understand the sentence "Time flies like an arrow"?
* Enormous flexibility
    * Hardly any constraints on the form of inputs and outputs. Unlike linear models or trees that needs a tabular data.
    * You can add constraints and change any components you want. The same network (or the produce of one) can be used for multiple tasks that appear to unrelated.
    * e.g. `Transfer Learning` with BERT. This techniques beat the STOA (state of the art) result on 13 NLP challenges back in 2018.
* Can be easily used with Online Learning


Cons
* Time/Resource consuming
    * Takes hours, days, months to train
    * Takes GPUs/TPUs $
* Very sensitive to experimental setup
    * initialization
    * hyperparameter tuning
    * component selection
    * feature distribution

Summary, deep learning has more `art` part than applying a gradient boosting tree, so whenever you can, fit a `gbm` when your dl modeling is training.


### Architectures
---

Since DL is good at addressing unstructured tasks, many architectures were evolved over the years. Here are a few important ones:

Major families
* MLP - Multi-layer perceptron
* CNN - Convolutional Neural Network
* RNN - Recurrent Neural Network

Any more advanced ones
* Transformers
* GNN - Graphical Neural Network
* GAN - Generative Adversarial Networks

Every family have some powerful decendents. Also, the families are not mutually exclusive.

`It is better to view the architectures based on the problems they are solving, instead of the model architectures. Each model architectures are defined by some core attribute. Just need to match the attribute to the correct task.`

#### Task Perspective
* tables
    * popular architectures: MLP, TabNet
* sequence modeling 
    * data when there is a sequence ordering. e.g. stock market price.
    * popular architectures: Transformer, CNN, LSTM (RNN)
* 2d inputs - images
    * when data 'should' be viewed in 2d
    * popular architectures: CNN, Transformer
* generative tasks
    * Autoencoders
    * GANs
* more advanced topics...
    

### Important Topics
---

As the field evolve by break-throughs. Please know the terminologies: 
* Fully connected/dense layers
* Convolutions
* Scalar, Vector, Matrix, Tensor
* Activation Functions
    * ReLU
    * Sigmoid/logis
* Optimizers
    * Adam
    * SGD
* Loss Function
    * MSE
    * Cross Entropy
* Training Techniques
    * Dropout
    * Batch Normalization
    * Embedding layer - for categorical variables
    * Residual Connection

### Popular Frameworks
---

Frameworks
* [Tensorflow/Keras](https://www.tensorflow.org/tutorials/keras/classification)
* [PyTorch](https://pytorch.org/tutorials/)


### Practical Suggestions
---

How to Learn/Practise
* Pick a framework and COPY the official examples
* Replicate other people's networks

Gist on model fitting
* Big the largest model you -- overfit as much as you can -- then regularize!
* [A Recipe for Training NN](http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines)