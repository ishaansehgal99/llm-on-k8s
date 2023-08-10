What makes llama 2 unique from traditional transformer based architecture? 

-> Increased context length, context window for llama 2 model is doubled in size from 2048 to 4096 tokens. This allows model to process more information and better understand long sequences of text. 

-> Grouped Query Attention (GQA) Llama 2 model uses GQA to improve effciency of the attention mechanism (attention is the most time consuming part of transformer arch). GQA groups similiar tokens together and attends to them as a single unit (uses the same Key, Query, Value) for them. This reduces number of computations that need to be performed which makes the model faster and more efficient. 

-> RMS Pre-Normalization. Transformers can use normalization to improve stability and performance of the network. LLama 2 uses RMS pre-normalization of the input features to normalize them. More detailed 
- Input to each transformer sub-layer is a sequence of vectors 
- Normalizes calculates the mean and std of each vector in the sequence 
- Vectors are normalized by subtracting mean and dividing by std 
- Normalized vectors are then passed to the transformer sublayer 


-> Rotary Positional Embeddings (RoPE). Traditional transformer uses sinusoidal waves to calculate each position embedding values. RoPE uses rotation matrices to calculate positional embeddings. 


-> SwiGLU activation function


