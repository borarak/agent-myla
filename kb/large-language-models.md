---
title: Large Language Models (LLMs)
concept_id: large-language-models
status: deep
prereqs: 
sources: https://towardsdatascience.com/understanding-large-language-models-lms-8c561b92ab5e,https://arxiv.org/abs/1706.03762,https://jalammar.github.io/illustrated-transformer/,https://github.com/ml6team/QLoRA,https://arxiv.org/abs/1810.04805
---

## Intuition

Imagine you’re having a conversation with your friend who has read an immense number of books and articles. No matter what subject you bring up, they seem to know something about it — responsive and engaging, they provide coherent and insightful replies. This is somewhat analogous to how Large Language Models (LLMs) operate.

### What Are LLMs?

Large Language Models are computer programs designed to understand and generate human-like text. They are trained on massive datasets that encompass a wide range of topics, styles, and contexts, allowing them to mimic the way humans communicate. Think of LLMs as sophisticated "text prediction" engines. They don’t simply regurgitate information; instead, they create sentences based on the immense variety they've seen during training, forming coherent and contextually relevant outputs.

### Why Do We Need LLMs?

The need for LLMs arises from several challenges in natural language processing (NLP). Here are a few key reasons:

1. **Human-like Interaction**: As our world becomes more digital, there’s a growing demand for machines that can interact with humans not just in a robotic or scripted way, but naturally. Whether it's customer service chatbots or virtual assistants, the ability to understand and respond intelligently is crucial.

2. **Information Retrieval**: In an age overflowing with information, sometimes it’s not about having data but about deriving insights from it in a conversational format. LLMs help summarize content, generate ideas, and provide contextually relevant replies, making human interactions with data more efficient and productive.

3. **Creative Assistance**: LLMs can generate poems, stories, and even code. This opens up new avenues for creativity, providing inspiration and aiding individuals in their creative processes.

4. **Personalization**: As LLMs learn from the style and preferences of individual users, they can tailor their responses, making conversations more engaging and relevant.

### The Mechanics Behind LLMs

At a high level, LLMs utilize a combination of statistical patterns and machine learning principles to craft their responses. Picture it like this: when you predict what a friend might say next based on their previous comments, you're using your experience and knowledge. LLMs do something similar, but on a much larger scale, drawing from patterns recognized in their training data.

- **Training Data**: They’re trained on diverse internet text — books, articles, social media posts, and more, which contributes to their vast knowledge base.

- **Patterns and Associations**: They look for patterns in how words and phrases relate to one another in context. For instance, if the phrase "baking a cake" appears frequently with words like "flour," "oven," and "recipe," the model learns to associate those words together in similar contexts.

- **Fine-Tuning**: This is where techniques like QLoRA come into play. Fine-tuning helps to adapt an LLM to perform better on specific tasks or for particular audiences. Think of it as teaching a well-read friend to be an expert on a specific niche topic.

### What Problems Do LLMs Solve?

- **Language Understanding**: They break down the barriers of linguistic complexity, helping machines understand and generate natural language.
- **Scalability**: With LLMs, you can scale interactions across thousands or millions of users simultaneously, a feat difficult to achieve with traditional methods.
- **Context Awareness**: They can keep track of context over a conversation, making them particularly useful for applications requiring ongoing dialogue.

In essence, Large Language Models exist not only to facilitate human-machine interactions but also to enhance creativity, provide personalized experiences, and efficiently process information in our increasingly text-rich environment. As we continue to develop methods like QLoRA, our ability to leverage these models will expand even further, allowing for more refined and effective applications in the realms of AI and beyond.

## Mechanism

Large Language Models (LLMs) are a class of machine learning models designed for understanding and generating human language. They leverage neural network architectures, particularly transformers, to process input text and produce relevant outputs. Here's a breakdown of how LLMs work:

### 1. Architecture
The fundamental building block of LLMs is the transformer architecture, which consists of two main components: the encoder and the decoder. However, in many state-of-the-art LLMs, especially those focused on text generation, only the decoder is utilized.

### 2. Tokenization
Before language processing can occur, the input text must be tokenized. This involves splitting the text into smaller units (tokens), such as words or subwords, that can be represented numerically. Tokenization allows for the handling of diverse vocabularies and different languages.

### 3. Input Representations
Once tokenized, tokens are converted into embeddings. Embeddings are dense vectors that provide a continuous representation of discrete tokens. LLMs typically utilize a learned embedding layer that transforms each token into a high-dimensional space.

### 4. Self-Attention Mechanism
The core of the transformer model is the self-attention mechanism. It helps the model determine the importance of each token in the context of other tokens. By calculating attention scores, the model can weigh the influence of surrounding tokens on the representation of a specific token. 

The self-attention mechanism is defined mathematically by:
\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]  
where Q, K, and V represent the queries, keys, and values, respectively, and \(d_k\) is the dimensionality of the key vectors.

### 5. Positional Encoding
Since transformers do not inherently understand the sequence of the input tokens, positional encodings are added to token embeddings to introduce information about the order of tokens. This allows the model to take into account the position of words within sentences.

### 6. Training Objective
LLMs are typically trained using unsupervised learning approaches, often with a language modeling objective. The common objective is to maximize the likelihood of predicting the next word in a sentence, given previous words.

The loss function often used is the cross-entropy loss, which measures the difference between the predicted probabilities and the actual tokens.

### 7. Fine-Tuning
After pre-training on large text corpora, LLMs are often fine-tuned on specific tasks or datasets using techniques such as QLoRA. Fine-tuning adjusts the model weights optimally for specialized tasks, improving performance.

### Computed Example
To illustrate the basic mechanics, let's compute the dot product in the attention mechanism using NumPy. Here’s how you can implement it:

```python
import numpy as np

# Sample query (Q), key (K), and value (V) matrices
Q = np.array([[1, 0], [0, 1]])  # 2 queries
K = np.array([[1, 1], [0, 1]])  # 2 keys
V = np.array([[1, 2], [3, 4]])  # 2 values

d_k = Q.shape[1]  # Dimensionality of the key vectors

# Calculate the Attention scores
scores = np.dot(Q, K.T) / np.sqrt(d_k)
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

# Compute the output of the attention mechanism
output = np.dot(attention_weights, V)
print("Attention Weights:", attention_weights)
print("Output:", output)
```

### Output Explanation
In this example:
- The attention scores are computed based on the dot product of queries and keys, scaled by the square root of the dimension of keys.
- Softmax is applied to convert raw scores into probabilities (attention weights).
- Finally, these weights are applied to the values to produce the output of the attention mechanism.

This encapsulates the core mechanism of LLMs and demonstrates how self-attention operates within the model. The output will provide insight into the model's focus on various parts of the input text.

## Formalism

### Definition
A Large Language Model (LLM) is a type of artificial intelligence model designed to understand, generate, and manipulate human language. Formally, an LLM can be defined as a function  
\[  L: \mathcal{X} \rightarrow \mathcal{Y} \]  
where:
- \(\mathcal{X}\) is the input space consisting of sequences of tokens (words, subwords, etc.)  
- \(\mathcal{Y}\) is the output space, which can be sequences of tokens again, or other outputs based on the task (e.g., classifications).

### Architecture
LLMs are often based on the Transformer architecture, which consists of an encoder-decoder structure. Formally, the processes involved can be expressed as:

1. **Encoding**: The input sequence \(x = (x_1, x_2, \ldots, x_n)\) is transformed into a sequence of embeddings \(E(x) = (E(x_1), E(x_2), \ldots, E(x_n))\) using an embedding matrix \(W_e\).  
2. **Self-Attention Mechanism**: The scaled dot-product attention mechanism is computed as follows:  
 \[  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V \]  
   where \(Q\), \(K\), and \(V\) are the query, key, and value matrices, and \(d_k\) is the dimension of the key vectors.
3. **Feed-Forward Networks**: The output of the attention layer is then passed through a feed-forward neural network denoted as \(FFN(x) = \sigma(W_2 \sigma(W_1 x + b_1) + b_2)\), where \(W_1, W_2\) are weight matrices and \(b_1, b_2\) are bias vectors.
4. **Layer Normalization and Residual Connection**: After each attention and feed-forward operation, layer normalization and residual connections are applied to stabilize training:
   \[  \text{LayerNorm}(x + \text{Sublayer}(x)) \]

### Training Objective
The objective of training LLMs typically involves minimizing a loss function for a given task. For language modeling, the loss function is usually the negative log-likelihood given by:
 \[  L = -\sum_{t=1}^{T} \log P(x_t | x_{<t}) \]  
where \(x_t\) is the token at time step \(t\), and \(P(x_t | x_{<t})\) is the probability of \(x_t\) given all previous tokens in the sequence.

### Theorem: Universality of LLMs
**Theorem 1**: Every function that maps sequences of tokens to outputs can be approximated arbitrarily closely by a sufficiently large language model if it has enough parameters.  

**Proof**: The proof is based on the universal approximation theorem for neural networks. As transformers can be thought of as a type of neural networks with sufficient capacity, for any target function \(f: \mathcal{X} \rightarrow \mathcal{Y}\), given \(\epsilon > 0\), there exists a model \(L\) with parameters \(\theta\) such that:
 \[  ||L(x; \theta) - f(x)|| < \epsilon \]  
for all \(x \in \mathcal{X}\).

### Conclusion
Large Language Models play a crucial role in natural language processing and are foundational for implementing fine-tuning techniques, such as QLoRA. Understanding their mechanics and theoretical foundations allows engineers and researchers to leverage these models effectively in various applications.

**Computed example — Calculate negative log-likelihood for a sequence of tokens**
Inputs: {'tokens': ['the', 'cat', 'sat'], 'probabilities': [0.1, 0.7, 0.2]}  
Output: L = - (log(0.1) + log(0.7) + log(0.2)) = 4.0311  
Tool: numpy