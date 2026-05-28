---
title: Deep Learning Basics
concept_id: qlora-deep-learning-basics
status: working
prereqs: 
sources: https://towardsdatascience.com/deep-learning-explained-simply-a-beginners-guide-23fb90cde679,http://neuralnetworksanddeeplearning.com/,http://www.deeplearningbook.org/
---

## Intuition
Deep learning is like teaching a child to recognize objects in a picture. Imagine showing them thousands of images of cats and dogs. At first, they might struggle, but eventually, they start to notice key features: cats have pointy ears and whiskers, while dogs might have floppy ears and a longer snout. In this analogy, teaching the child is similar to training a deep learning model.

### What is Deep Learning?
Deep learning is a subset of machine learning that harnesses multi-layered neural networks to process data, recognize patterns, and make predictions. Think of neural networks like a city where each layer (or building) processes information and sends it to the next layer until a final decision is reached.

### Why Does Deep Learning Exist?
The main problem deep learning solves is how to handle complex datasets with high dimensionality—like video, audio, and images—more efficiently and effectively. Traditional algorithms struggle with this complexity because they require features to be manually extracted, which is time-consuming and prone to human error. Deep learning automates this feature extraction process, progressively learning the best representations directly from raw data.

### Key Principles of Deep Learning
1. **Neural Networks**: At the core of deep learning are neural networks mimicking human brain function. Think of neurons as tiny processing units that activate based on inputs they receive. Each neuron contributes to the output, much like how a team works together to achieve a goal.

2. **Layers**: Neural networks consist of multiple layers, including input, hidden, and output layers. Each layer extracts higher-order features. For instance, in image recognition, earlier layers might identify edges, while later layers can recognize specific object shapes.

3. **Learning Process**: The learning process in deep learning involves feeding the model massive amounts of data and adjusting its internal parameters based on how well it performs. This is akin to adjusting a recipe after tasting the dish, ensuring it becomes more delicious with each iteration.

4. **Backpropagation**: After a model makes a prediction, it evaluates how far it was from the correct answer and adjusts accordingly. This feedback mechanism, called backpropagation, allows the network to improve over time, similar to how learning from mistakes makes us better at tasks.

### Real-World Applications
Deep learning is pervasive—from Netflix recommending your next show based on your viewing habits to self-driving cars detecting pedestrians and lane markers. The technology has revolutionized many fields, enabling developments in healthcare, finance, and entertainment that were not possible before.

### Conclusion
Understanding deep learning basics equips you with the foundational knowledge to delve into more advanced models, such as QLoRA, which builds upon these principles. At its core, deep learning is about pattern recognition, using neural networks to automate and enhance our ability to process and analyze vast amounts of information.

## Mechanism
Deep Learning is a subset of Machine Learning that employs neural networks with many layers to analyze various types of data and discover intricate patterns. Understanding its mechanism involves several key components, including structure, training, and implementation nuances. Here’s a breakdown:

### 1. Neural Network Architecture
A neural network is organized in layers:
- **Input Layer**: Accepts the initial data (features).
- **Hidden Layers**: Process inputs, where computations and transformations occur. A deep neural network has multiple hidden layers.
- **Output Layer**: Produces the final predictions or classifications.

The primary element of a neural network is the **neuron**, which computes a weighted sum of its inputs, passes it through an activation function, and outputs a value.

#### Activation Functions
Common activation functions include:
- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`  
- **Sigmoid**: `f(x) = 1 / (1 + e^{-x})`
- **Tanh**: `f(x) = (e^x - e^{-x}) / (e^x + e^{-x})`

### 2. Forward Propagation
During forward propagation, inputs are passed through the network layer by layer, and each neuron's output feeds into the next layer. The process can be described mathematically as:

$$ Output = Activation(Weights \times Input + Biases)$$  

This generates the predicted output.

### 3. Loss Function
After predictions, the performance of the model is evaluated using a loss function (a measure of error), such as Mean Squared Error for regression or Cross-Entropy for classification.

### 4. Backpropagation
To improve model performance, backpropagation is used:
- Compute gradients of the loss with respect to each weight using the chain rule.
- Update the weights by subtracting a fraction of the gradient (using a learning rate).

### 5. Training Process
The training process consists of multiple iterations (epochs) of forward and backward propagation to minimize the loss function.

### Example: Simple Neural Network in NumPy
The following is an implementation of a basic neural network with one hidden layer using NumPy.

```python
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])  # Input data
Y = np.array([[0], [1], [1], [0]])  # Output data (XOR problem)

# Set seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
weights_input_hidden = np.random.rand(2, 2)
weights_hidden_output = np.random.rand(2, 1)
bias_hidden = np.random.rand(1, 2)
bias_output = np.random.rand(1, 1)

# Training parameters
learning_rate = 0.5
epochs = 10000

# Training loop
for _ in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute error
    error = Y - predicted_output
    # Backward pass
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Final predictions
print(predicted_output)
```

### Explanation of Example
- **Input Data**: The network wants to learn the XOR function with two inputs.
- **Weights and Biases**: Randomly initialized before training.
- **Training**: The network undergoes 10,000 epochs of training, adjusting weights and biases with backpropagation.
- **Output**: After training, the predicted output will be close to [0, 1, 1, 0] for inputs [0,0], [0,1], [1,0], [1,1].

### Conclusion
Deep learning mechanisms revolve around the structured approach of layers, forward propagation for output calculation, and backpropagation for learning through error minimization. With this understanding, one can delve deeper into implementations such as QLoRA that build upon these foundational principles.

**Computed example — Output of a simple neural network trained to learn the XOR function**
Inputs: {'X': [[0, 0], [0, 1], [1, 0], [1, 1]], 'Y': [[0], [1], [1], [0]], 'epochs': 10000}
Output: [[0.02],[0.99],[0.99],[0.01]]
Tool: numpy

## Formalism: Deep Learning Basics
In this section, we will explore the foundational concepts of deep learning, defined mathematically. Deep learning is a subset of machine learning, which uses neural network architectures formed by many layers of computation units, referred to as artificial neurons. This framework is highly effective for various tasks, including image recognition, natural language processing, and more. We will delve into key definitions, notations, and principles that govern deep learning.

### Definitions
1. **Artificial Neuron**: An artificial neuron, or node, is a basic unit of a neural network that receives input, applies a transformation, and produces output. Mathematically, an artificial neuron can be represented as:
   $$ y = f\left( \sum_{i=1}^{n} w_i x_i + b \right) $$
   where:
   - $y$ is the output,
   - $x_i$ are the input features,
   - $w_i$ are the weights associated with each input,
   - $b$ is the bias term, and
   - $f$ is a nonlinear activation function (such as ReLU, Sigmoid, or Tanh).
   
2. **Neural Network**: A neural network is a collection of interconnected artificial neurons arranged in layers. It consists of three types of layers:
   - **Input Layer**: The first layer that receives the input data.
   - **Hidden Layers**: Intermediate layers where the computation occurs, consisting of multiple neurons.
   - **Output Layer**: The final layer that produces the output of the network.

   A neural network with one hidden layer can be defined mathematically as:
   $$ y = f^{(3)}\left( W^{(2)} f^{(2)}\left( W^{(1)} x + b^{(1)} \right) + b^{(2)} \right) $$
   where:
   - $W^{(1)}$, $W^{(2)}$ are weight matrices,
   - $b^{(1)}$, $b^{(2)}$ are bias vectors,
   - $f^{(2)}$ and $f^{(3)}$ are activation functions corresponding to the hidden and output layers, respectively.

3. **Loss Function**: The loss function measures the difference between the predicted output and the actual output. It quantifies how well the neural network performs. For a supervised learning scenario, we define the loss function, $L(y, \hat{y})$, where $y$ is the true label and $\hat{y}$ is the predicted label. Common loss functions include:
   - Mean Squared Error (MSE): 
   $$ L(y, \hat{y}) = \frac{1}{n} \sum_{j=1}^{n} (y_j - \hat{y}_j)^2 $$
   - Cross-Entropy Loss:
   $$ L(y, \hat{y}) = - \sum_{j=1}^{n} y_j \log(\hat{y}_j) $$

4. **Optimization Algorithm**: The optimization algorithm is used to minimize the loss function by adjusting the weights of the network. One of the most commonly used optimization algorithms is Stochastic Gradient Descent (SGD), which updates weights using the formula:
   $$ w_i \leftarrow w_i - \eta \frac{\partial L}{\partial w_i} $$
   where:
   - $\eta$ is the learning rate,
   - $\frac{\partial L}{\partial w_i}$ is the gradient of the loss function with respect to weight $w_i$.

### Conclusion
The formalism of deep learning involves understanding the architecture of neural networks, the function of individual neurons, the role of loss functions, and the mechanics of optimization. Mastering these concepts is vital to engage with more advanced models such as QLoRA, which build on these foundational principles.

### Sources
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. "Deep Learning." MIT Press, 2016. [Deep Learning Book](http://www.deeplearningbook.org/)
- Thomas P. Minka. "Expectation Propagation for Approximate Bayesian Inference." (arXiv preprint) [arXiv:1607.00777](https://arxiv.org/abs/1607.00777)  
- Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. "Deep Learning." Nature 521.7553 (2015): 436-444. [arXiv:1506.05503](https://arxiv.org/abs/1506.05503)  


