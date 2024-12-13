!pip install einops
class TS_AttentionModule(layers.Layer):
    def __init__(self, emb_size, num_heads, dropout):
        super(TS_AttentionModule, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Linear layers for queries, keys, and values
        self.query_dense = layers.Dense(emb_size)
        self.key_dense = layers.Dense(emb_size)
        self.value_dense = layers.Dense(emb_size)
        self.dropout_layer = layers.Dropout(dropout)
        
        # Linear projection after attention output
        self.projection = layers.Dense(emb_size)

    def call(self, query, key, value, mask=None):
        # Dense layers for queries, keys, and values
        queries = self.query_dense(query)
        keys = self.key_dense(key)
        values = self.value_dense(value)
        
        # Reshape for multi-head attention
        batch_size = tf.shape(queries)[0]
        seq_len = tf.shape(queries)[1]
        head_dim = self.emb_size // self.num_heads
        
        # Reshaping queries, keys, and values for multiple heads
        queries = tf.reshape(queries, (batch_size, seq_len, self.num_heads, head_dim))
        keys = tf.reshape(keys, (batch_size, seq_len, self.num_heads, head_dim))
        values = tf.reshape(values, (batch_size, seq_len, self.num_heads, head_dim))

        # Transpose to prepare for scaled dot-product attention
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(queries, keys, transpose_b=True) # energy
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(head_dim, tf.float32))
        
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Calculate attention output
        attention_output = tf.matmul(attention_weights, values)
        
        # Transpose and reshape back to original form
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.emb_size))
        
        # Final projection to combine heads
        output = self.projection(attention_output)
        
        return output

def TS_ResidualAdd(x, fn):
    """
    Residual connection followed by function application.
    
    Parameters:
        x (tf.Tensor): Input tensor.
        fn (function): A function to apply to the input tensor.
        
    Returns:
        tf.Tensor: Output tensor after residual connection and function application.
    """
    res = x
    x = fn(x)
    return res + x

def TS_FeedForwardBlock(x, emb_size, expansion, drop_p):
    """
    Feed-forward block with linear layers.
    
    Args:
        x (tf.Tensor): Input tensor.
        emb_size (int): Embedding size.
        expansion (int): Expansion factor for the feed-forward layer.
        drop_p (float): Dropout probability.
    
    Returns:
        tf.Tensor: Output tensor after feed-forward processing.
    """
    # First dense layer with expansion
    x = layers.Dense(expansion * emb_size)(x)
    
    # GELU activation wrapped in a Keras layer
    x = layers.Activation('gelu')(x)
    
    # Dropout wrapped in a Keras layer
    x = layers.Dropout(drop_p)(x)
    
    # Second dense layer to map back to embedding size
    x = layers.Dense(emb_size)(x)
    
    return x
from einops import rearrange

def attention_block(in_layer, freq_features, emb_size=32, num_heads=4, dropout=0.3, ratio=8, 
                    residual=True, apply_to_input=True, forward_expansion=4, forward_drop_p=0.5, 
                    **kwargs):
    
    # Positional encoding initialization (can be trainable or fixed)
    seq_len = in_layer.shape[1]
    pos_embedding_in = layers.Embedding(input_dim=seq_len, output_dim=emb_size)
    pos_embedding_freq = layers.Embedding(input_dim=freq_features.shape[1], output_dim=emb_size)
    
    # Step 1: Layer normalization for stability
    layer_norm = layers.LayerNormalization(epsilon=1e-6)
    in_layer_norm = layer_norm(in_layer)
    freq_features_norm = layer_norm(freq_features)

    # Step 2: Add positional encoding to both time-series and frequency features
    pos_indices_in = tf.range(start=0, limit=seq_len, delta=1)
    pos_indices_freq = tf.range(start=0, limit=freq_features.shape[1], delta=1)

    in_layer_pos_encoded = in_layer_norm + pos_embedding_in(pos_indices_in)
    freq_features_pos_encoded = freq_features_norm + pos_embedding_freq(pos_indices_freq)

    # Step 3: Project and Reshape Frequency Features before Cross-Attention
    freq_features_projected = layers.Dense(in_layer.shape[1])(freq_features_pos_encoded)
    freq_features_norm = layers.LayerNormalization(epsilon=1e-6)(freq_features_projected)
    freq_features_norm = layers.Reshape((in_layer.shape[1], -1))(freq_features_norm)

    # Step 4: Use TS_AttentionModule for Cross-Attention with freq_features_norm as key and value
    cross_attention_output = TS_AttentionModule(emb_size, num_heads, dropout)(
        query=in_layer_pos_encoded, key=freq_features_norm, value=in_layer_pos_encoded)

    # Step 5: Residual Connection after Cross-Attention using TS_ResidualAdd
    if residual:
        cross_attention_output = TS_ResidualAdd(
            in_layer_pos_encoded, lambda x: TS_AttentionModule(emb_size, num_heads, dropout)(
                x, key=freq_features_norm, value=in_layer_pos_encoded))

    # Step 6: Apply the Feedforward Block using TS_FeedForwardBlock
    feedforward_output = TS_FeedForwardBlock(cross_attention_output, emb_size, forward_expansion, forward_drop_p)

    # Step 7: Final residual connection (optional)
    if residual:
        final_output = TS_ResidualAdd(cross_attention_output, lambda x: TS_FeedForwardBlock(x, emb_size, forward_expansion, forward_drop_p))
    else:
        final_output = feedforward_output

    return final_output
