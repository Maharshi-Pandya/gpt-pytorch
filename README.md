# gpt-pytorch

a simple implementation of the GPT-1 paper in pytorch.

here's the [link to the paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


### todo

- Our model largely follows the original transformer work. We trained a
12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states.

- We used the Adam optimization scheme with a max learning rate of 2.5e-4.

- The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.

- We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens. Since layernorm is used extensively throughout the model, a simple weight initialization of N(0, 0.02) was sufficient. We used a bytepair encoding (BPE) vocabulary with 40,000 merges and residual, embedding, and attention dropouts with a rate of 0.1 for regularization. 

- We also employed a modified version of L2 regularization, with w = 0.01 on all non bias or gain weights. For the activation function, we used the Gaussian Error Linear Unit (GELU). 

- We used learned position embeddings instead of the sinusoidal version proposed in the original work.

- We use the ftfy library2 to clean the raw text in BooksCorpus, standardize some punctuation and whitespace, and use the spaCy tokenizer.
