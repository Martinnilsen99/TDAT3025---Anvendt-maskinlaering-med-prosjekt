### Convolutional Neural Networks have several types of layers:

    Convolutional layer
    * a “filter” passes over the image, scanning a few pixels at a time and creating a feature map that predicts the class to which each feature belongs.

    Pooling layer (downsampling)
    * reduces the amount of information in each feature obtained in the convolutional layer while maintaining the most important information (there are usually several rounds of convolution and pooling).

    Fully connected input layer (flatten)
    * takes the output of the previous layers, “flattens” them and turns them into a single vector that can be an input for the next stage.

    The first fully connected layer
    * takes the inputs from the feature analysis and applies weights to predict the correct label.

    Fully connected output layer
    * gives the final probabilities for each label.

#### Dense and fully-connected
Dense and fully connected is the same thing; A linear operation in which every input is connected to every output by a weight (so there are n_inputs * n_outputs weights - which can be a lot!). Generally followed by a non-linear activation function.


### ReLU
###### What it is
The ReLU function is another non-linear activation function that has gained popularity in the deep learning domain. ReLU stands for Rectified Linear Unit. The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time

###### Where to use it
PyTorch's own documentation states that it should be used in the forwarding-method as:

```py
x = self.pool(F.relu(self.conv1(x)))
```
convolutional layer -> ReLu -> pool

*Note that if you would like to include dropout this can also be included in this line (p=0.1 or p=0.2, not p=0.5 as if included on the fully connected layers), as written below. The order will then be:
conv -> ReLu -> dropout -> pool

### Dropout
###### What it is
Deep learning neural networks are likely to quickly overfit a training dataset with few examples.

Ensembles of neural networks with different model configurations are known to reduce overfitting, but require the additional computational expense of training and maintaining multiple models.

A single model can be used to simulate having a large number of different network architectures by randomly dropping out nodes during training. This is called dropout and offers a very computationally cheap and remarkably effective regularization method to reduce overfitting and improve generalization error in deep neural networks of all kinds.

(<https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/>)


###### Where to use it
In the original paper that proposed dropout layers, by Hinton (2012), dropout (with p=0.5) was used on each of the fully connected (dense) layers before the output; it was not used on the convolutional layers. This became the most commonly used configuration.

More recent research has shown some value in applying dropout also to convolutional layers, although at much lower levels: p=0.1 or 0.2. Dropout was used after the activation function of each convolutional layer: CONV->RELU->DROP.

(<https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network>)