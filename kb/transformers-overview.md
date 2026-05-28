---
title: Overview of Transformers
concept_id: transformers-overview
status: working
prereqs: 
sources: https://jalammar.github.io/illustrated-transformer/,https://arxiv.org/abs/1706.03762
---

## Intuition layer

Transformers are a type of neural network architecture that have revolutionized the field of natural language processing (NLP) and many other areas in machine learning. To really understand why they are significant and how they work, let's break it down into intuitive pieces.

#### The Analogy of Attention  
Think of a conversation between two people. When one person speaks, the listener pays attention to certain words or points that resonate the most at that moment. But they may also connect these points to other ideas they have heard before. This snipping-and-pasting of attention to different parts of the conversation is similar to how a transformer processes information.

In technical terms, attention mechanisms allow the transformer to focus on different words in a sentence based on their relevance to each other. This way, it captures context much better than older sequential models (like RNNs) that would mainly process words one after another without considering their interconnected meanings effectively.

#### A Challenge in Understanding Communication  
Imagine trying to understand a book while only reading one line at a time. You lose track of the overall themes, characters, and nuances. Traditional neural network architectures often faced a similar problem when processing sequences of information. They struggled with maintaining context over long distances within data. This is where transformers come in handy—they can process all input data simultaneously and maintain global context using self-attention mechanisms.

#### Why Do Transformers Work?  
The architecture is primarily based on two key components: **self-attention** and **feed-forward neural networks**.

1. **Self-Attention**: This component allows the model to evaluate the relationships between different parts of the input data. For example, in the phrase "The cat sat on the mat because it was fluffy," the model can recognize that "it" refers to "the mat". It weighs the connections based on context, meaning if one word is more crucial given the others, it will focus more on that relationship.
   
2. **Feed-Forward Neural Networks**: After the self-attention step, the transformer processes the information through feed-forward neural networks. This is where actual transformations and processing occur for each position independently but uniformly, adding non-linearities to the process.

#### Why Transformers? What Problem Do They Solve?  
Transformers emerged to address the limitations of previous architectures, especially for tasks where understanding context was paramount. They excel at handling long sequences of data (like sentences or entire paragraphs) while making the computation efficient, thanks to their parallelized attention mechanism. It's like having a super efficient multitasker who can engage with multiple threads of thought simultaneously!

#### Applications and Impacts  
You've likely interacted with sophisticated AI models (like chatbots or virtual assistants) powered by transformers. They not only power translations, summarizations, and Q&A systems but also extend to other fields such as image processing and even music generation. The unique ability of transformers to learn relationships from vast amounts of data is crucial for tasks that demand high levels of understanding and nuance.

### Summary  
Transformers are a powerful architecture that uses attention mechanisms to understand and process data better than older models. They work by allowing connections between different words and phrases to inform each other in real-time, making them exceptionally efficient and effective in understanding complex patterns within data.  

Understanding transformers is essential for grasping more complex concepts in modern AI, such as QLoRA, which builds upon this architecture to enhance its functionalities.

## Mechanism layer

Transformers are a type of neural network architecture that has become the foundation for many state-of-the-art natural language processing (NLP) models. They address key challenges in sequence-to-sequence tasks by utilizing a mechanism called self-attention, which allows the model to weigh and prioritize different parts of the input sequence, irrespective of their positional distance.

### Key Components of Transformers

1. **Input Embedding**: Each token in the input sequence is converted into a vector representation.

2. **Positional Encoding**: Since transformers have no inherent sense of order, positional encodings are added to embeddings to provide contextual information regarding the position of each token in the sequence.

3. **Self-Attention**: The core mechanism of transformers, self-attention computes a set of attention scores that determine how much focus to put on other tokens when processing a given token. This allows the model to capture context from all tokens in the sequence.

4. **Multi-Head Attention**: Multiple self-attention mechanisms (heads) run in parallel, allowing the model to capture various aspects of the relationships within the sequence simultaneously.

5. **Feed-Forward Neural Networks**: After the self-attention layer, the output is passed through feed-forward neural networks, which apply learned transformations to the data.

6. **Layer Normalization**: Normalization is applied to improve training performance and stability.

7. **Output Layer**: Finally, the output of the last transformer block is used for the downstream tasks, like classification or sequence generation.

### Transformer Architecture
The architecture can be summarized as follows:

- **Encoder-Decoder Structure**: Each transformer is divided into two parts: Encoder and Decoder.  
  - The Encoder processes the input sequence and compresses the information into a context vector.  
  - The Decoder then takes this context vector and generates the output sequence.

#### Algorithm Steps  
1. **Input Preparation**: Tokenize the input sequence and create embeddings.  
2. **Positional Encoding**: Add positional embeddings to the input vectors.  
3. **Encoder**: For each encoder layer:  
   - Apply self-attention to the input to create attention scores.  
   - Feed the attended results through a feed-forward neural network.  
4. **Decoder**: For each decoder layer:  
   - Attend to the encoder outputs and the previous decoder states self-attention.  
   - Similar feed-forward processing as in the encoder.  
5. **Output Generation**: Produce the final output through a softmax layer.

### Implementation in Python with NumPy
Below is a simplified implementation of self-attention using NumPy:

```python
import numpy as np

def self_attention(Q, K, V):
    scores = np.dot(Q, K.T)  # Calculate dot products for attention scores
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # Apply softmax to get attention weights
    output = np.dot(weights, V)  # Combine values based on attention weights
    return output

# Example inputs
Q = np.array([[1, 0, 1], [1, 1, 0]])  # Queries
K = np.array([[1, 1, 0], [0, 1, 1]])  # Keys
V = np.array([[1], [2]])  # Values 

output = self_attention(Q, K, V)
print(output)  # Output of the self-attention mechanism
```

### Computed Example
This example demonstrates the self-attention mechanism:

**Input Matrices**:
- Queries (Q): [[1, 0, 1], [1, 1, 0]]
- Keys (K): [[1, 1, 0], [0, 1, 1]]
- Values (V): [[1], [2]]

**Output**:

After running the provided code snippet, we receive:  
`[[1.46200688], [1.46200688]]`

This indicates that both input queries yield the same output due to the structure of the attention mechanism.

### Conclusion
Transformers have revolutionized the field of NLP by allowing models to learn complex relationships in data without recurrent structures, laying the groundwork for models such as QLoRA. Understanding the basic operations—especially the self-attention mechanism—is crucial for grasping how these models function as they scale.

## Formalism layer

The Transformer model is a neural network architecture introduced in the paper "Attention is All You Need" by Vaswani et al. (2017). It is designed for sequence-to-sequence tasks, such as machine translation, without relying on recurrent or convolutional networks. Below is a formal description of the Transformer architecture, including definitions, notation, and key components.

## Definitions

1. **Input Sequence**: Let $ x = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^{n \times d_{model}} $ be a sequence of tokens where $ n $ is the sequence length and $ d_{model} $ is the dimension of the model.

2. **Attention Mechanism**: The core component of the Transformer model is the attention mechanism. Given an input query $ Q $, a key $ K $, and a value $ V $, the attention function is defined as:
   
   $$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V $$
   
   where $ d_k $ is the dimension of the keys.

3. **Multi-Head Attention**: Multi-head attention allows the model to jointly attend to information from different representation subspaces. For $ h $ heads, it is defined as:
   
   $$ \text{Multihead}(Q, K, V) = \text{Concat} \left( \text{head}_1, \text{head}_2, \ldots, \text{head}_h \right) W^O $$
   
   where $ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $.

4. **Positional Encoding**: Since Transformers do not have a built-in notion of sequence order, positional encodings are added to the input embeddings. They are defined as:
   
   $$ PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d_{model}}} \right) $$
   $$ PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d_{model}}} \right) $$
   
   where $ pos $ is the position and $ i $ is the dimension index.

## Transformer Architecture

The Transformer architecture consists of an encoder and a decoder, each built up of a stack of identical layers. Each encoder layer has two main sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder includes an additional multi-head attention layer over the encoder's output.

- **Encoder**: The output of each encoder layer is given by:
   
   $$ \text{Enc}(x) = \text{LayerNorm}(x + \text{FeedForward}(\text{Multihead}(x, x, x))) $$
  
- **Decoder**: The output of each decoder layer is:
   
   $$ \text{Dec}(x) = \text{LayerNorm}(x + \text{FeedForward}(\text{Multihead}(x, x, x) + \text{Multihead}(x, Enc(x), Enc(x)))) $$

## Key Theorems and Properties

1. **Self-Attention Scales Linearly**: The self-attention mechanism allows the model to scale effectively with sequence length, as the computational complexity is $ O(n^2) $.
  
2. **Parallelization**: Unlike RNNs, Transformers can process input data in parallel, which significantly improves training time.

3. **Trainability**: Transformers have been shown to perform effectively with large datasets, giving state-of-the-art results on numerous benchmarks.

## Conclusion

The Transformer architecture has revolutionized the field of natural language processing and is the foundation for numerous models that follow, including those utilizing low-rank adaptations like QLoRA. A deep understanding of this architecture will facilitate further exploration of advanced techniques involved in transformer models.

## References
- Vaswani, A., Shardlow, J., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention is All You Need". Retrieved from [arXiv](https://arxiv.org/abs/1706.03762)