by DeepLearningAI [(YouTube)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)


**July 17**
# 1. What is a neural network?

## 1.1. Neuron

The purpose of a neuron is to accept input, perform computations and return the result.

_For example_, A neuron can input the size of house from some sample data and predict the price of the house using that data. While it may be convenient to fit the data to a line (linear regression), most statistics follow a non-linear curve.

One non-linear relationship can be described by the **Rectified Linear Unit** (ReLU) function. This function returns a straight line when the input is positive, but zero otherwise.

$$
\mathrm{ReLU}(x)=\begin{cases}
x,\text{ if }x>0\\
0,\text{ otherwise}
\end{cases}
$$
# 1.2. Neural Network

A neural network comprises many neurons arranged in different layers. Some of these layers include:
- **Input** ($x$): All inputs to the network are provided in this layer.
- **Hidden**: These layers are responsible for all the computation in a neural network.
- **Output** ($y$): The result of all the computations from the hidden layer are returned in the output layer.

If every neuron in one layer is connected to every other neuron in its subsequent layer for every pair of consecutive layers in the network, the network is said to be _densely connected_.

If given given enough data about $x$ and $y$, a neural network is really good at determining accurate functions that map $x$ to $y$. This is discussed further in $\S$ 3.

**Source**: [Link](https://youtu.be/n1l-9lIMW7E?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
# 2. Supervised Learning

## 2.1. Types of Neural Network Architectures

Different neural network architectures have different applications. Some of them are listed as follows:
- **Standard neural networks (NNs)**: These are used for working with a finite set of data. They follow the same architecture as described in $\S$ 1.2.
- **Recurrent neural networks (RNNs)**: These are specialized for data that is input as a one-dimensional sequence, _i.e._ over a period of time, such as text or audio.
- **Convolutional neural networks (CNNs)**: These are used for inputting and performing computations on image data.
- **Hybrid networks**: These networks are used for complex data, such as image as well as audio or text.
## 2.2. Types of data

There are two kinds of data a neural network operates on, **Structured** and **Unstructured** data.

| **Structured data**                                                                                                                                  | **Unstructured data**                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Structured data refers to data that is organized in a tabular database. The features of a structured dataset can be easily understood by a computer. | Unstructured data refers to data whose features cannot be easily interpreted by a computer.                                                                                    |
| An example of a structured dataset is the analytics of an advert. In structured data, each parameter and statistic has a well-defined value.         | Examples of unstructured dataset include audio, image or text, where the meaningful features need to be extracted instead (such as audio cues, pixel data or words in a text). |
The objective of deep learning is to make the interpretation and computation of unstructured data easier with technologies such as _Speech Recognition_, _Image Recognition_ and _Natural Language Processing_.

**Source**: [Link](https://youtu.be/BYGpKPY9pO0?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)

# 3. Recent Improvements in Deep Learning

When we plot the performance of deep learning algorithms with respect to the amount of labelled data, we observe that traditional learning models such as Support Vector Machines and Logistic Regression tend to plateau (horizontal asymptote).

However, it is recently observed that larger neural networks, consisting of numerous neurons and connections, have shown a better performance increase with large amounts of data. Some of the reasons are discussed below:

## 3.1. Change from sigmoid to ReLU

Traditional algorithms used sigmoid as their activation functions. The problem with sigmoid is that at really large absolute values, the gradient (slope) of the sigmoid function is very close to zero, which causes the algorithm _Gradient descent_ to compute very slowly.

This issue is resolved since newer algorithms use the ReLU function (discussed in $\S$ 1.1.) which has a constant non-zero gradient for positive values.

**Source**: [Link](https://youtu.be/xflCLdJh0n0?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
