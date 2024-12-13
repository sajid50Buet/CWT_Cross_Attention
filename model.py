# Custom layer for squeezing
class SqueezeLayer(Layer):
    def __init__(self, axis):
        super(SqueezeLayer, self).__init__()
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)
    
class TransposeLayer(layers.Layer):
    def call(self, x):
        return tf.transpose(x, perm=[0, 1, 3, 2])  # Transpose to (batch_size, frequencies, channels, time)

def tf_conv_module(x, output_features=32, target_seq_len=None):
    """
    Modified Conv Module to produce features suitable for Key/Value in MultiHeadAttention.
    
    Parameters:
    - x: Input tensor (batch_size, frequencies, samples, channels)
    - output_features: Number of output features to match the query in MHA.
    - target_seq_len: The desired sequence length to match the query for MHA. If None, uses the frequency dimension size.
    
    Returns:
    - block1: Processed tensor ready for use as Key/Value in MultiHeadAttention.
    """
    weightDecay = 0.009
    maxNormValue = 0.6  # MaxNorm constraint value

    # Step 1: Transpose Layer to align channels (from input shape to (batch_size, frequencies, channels, time))
    x = TransposeLayer()(x)  # Output shape: (batch_size, frequencies, channels, time)
#     print("After TransposeLayer:", x.shape)

    # Step 2: First Conv2D Layer (Capture frequency-time patterns)
    x = layers.SeparableConv2D(16, (1, 10), padding='same', 
                           depthwise_regularizer=regularizers.l2(weightDecay),
                           pointwise_regularizer=regularizers.l2(weightDecay),
                           activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
#     print("After first Conv2D:", x.shape)

    # Step 3: Depthwise Conv2D Layer (Better capture of frequency patterns across channels)
    x = layers.DepthwiseConv2D(kernel_size=(1, 22), padding='same', 
                               depthwise_regularizer=regularizers.l2(weightDecay),
                               depthwise_constraint=MaxNorm(maxNormValue))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
#     print("After DepthwiseConv2D:", x.shape)
    
    # Step 4: Dropout for regularization
    x = layers.Dropout(0.5)(x)

    # Step 5: Pointwise Conv2D Layer (Increase feature complexity)
    x = layers.Conv2D(32, kernel_size=(1, 1), padding='same', 
                      kernel_regularizer=regularizers.l2(weightDecay),
                      kernel_constraint=MaxNorm(maxNormValue), activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
#     print("After Pointwise Conv2D:", x.shape)

    # Step 6: Second Conv2D Layer (Enhanced features across frequencies)
    x = layers.Conv2D(32, (4, 1), padding='same', 
                      kernel_regularizer=regularizers.l2(weightDecay),
                      kernel_constraint=MaxNorm(maxNormValue), activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
#     print("After second Conv2D:", x.shape)

    x = layers.AveragePooling2D(pool_size=(1, 18))(x)
    # print("After first AveragePooling2D:", x.shape)

    # Step 8: Dropout Layer for regularization
    x = layers.Dropout(0.6)(x)

    # Step 9: Final Conv Layer to reduce features to `output_features` (32)
    x = layers.Conv2D(output_features, (1, 1), padding='same', 
                      kernel_regularizer=regularizers.l2(weightDecay),
                      kernel_constraint=MaxNorm(maxNormValue), activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
#     print("After final Conv2D (output features):", x.shape)

    x = SqueezeLayer(axis=2)(x)
#     print("After Squeze:", x.shape)

    return x

def TS_Conv_block(input_layer, F1=4, kernLength=64, poolSize=7, D=2, in_chans=22, 
                  weightDecay=0.01, maxNorm=0.6, dropout=0.3):
    """ Conv_block with moderate kernel size variations and feature averaging using `layers`. """

    F2 = F1 * D
    
    # Block 1a: First Convolution with kernLength = 32
    block1a = layers.Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', 
                            kernel_regularizer=regularizers.L2(weightDecay),
                            kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_layer)
    block1a = layers.BatchNormalization(axis=-1)(block1a)

    # Block 1b: Convolution with kernLength + 16 = 48
    block1b = layers.Conv2D(F1, (kernLength + 16, 1), padding='same', data_format='channels_last', 
                            kernel_regularizer=regularizers.L2(weightDecay),
                            kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_layer)
    block1b = layers.BatchNormalization(axis=-1)(block1b)

    # Block 1c: Convolution with kernLength - 16 = 16
    block1c = layers.Conv2D(F1, (kernLength - 16, 1), padding='same', data_format='channels_last', 
                            kernel_regularizer=regularizers.L2(weightDecay),
                            kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_layer)
    block1c = layers.BatchNormalization(axis=-1)(block1c)

    # Averaging the outputs from different kernel sizes
    block1 = layers.Average()([block1a, block1b, block1c])

    # Block 2: Depthwise Convolution
    block2 = layers.DepthwiseConv2D((1, in_chans), depth_multiplier=D, data_format='channels_last',
                                    depthwise_regularizer=regularizers.L2(weightDecay),
                                    depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(block1)
    block2 = layers.BatchNormalization(axis=-1)(block2)
    block2 = layers.Activation('elu')(block2)
    
    # Adjusted Pooling to retain more spatial information
    block2 = layers.AveragePooling2D((6, 1), data_format='channels_last')(block2)  # Reduce pooling size to (6,1)
    block2 = layers.Dropout(dropout)(block2)
    
    # Block 3: Final Convolution
    block3 = layers.Conv2D(F2, (16, 1), padding='same', data_format='channels_last',
                           kernel_regularizer=regularizers.L2(weightDecay),
                           kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(block2)
    block3 = layers.BatchNormalization(axis=-1)(block3)
    block3 = layers.Activation('elu')(block3)
    
    # Final Pooling
    block3 = layers.AveragePooling2D((7, 1), data_format='channels_last')(block3)
    block3 = layers.Dropout(dropout)(block3)
    
    return block3

class GatedLinearUnit(Layer):
    def __init__(self, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
    
    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)  # Split into two equal parts
        return x1 * tf.sigmoid(x2)  # Apply gating

def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, 
               weightDecay=0.009, maxNorm=0.6):
    """ TCN block with GLU and optimized dropout """
    
    # Initial Conv1D block with GLU
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = SpatialDropout1D(dropout)(block)
    
    block = GatedLinearUnit()(block)  # Apply GLU as a layer

    # Second Conv1D block with GLU
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = SpatialDropout1D(dropout)(block)

    # Residual connection
    if input_dimension != filters:
        conv = Conv1D(filters, kernel_size=1, activation='linear',
                      kernel_regularizer=L2(weightDecay),
                      kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                      padding='same')(input_layer)
        conv = BatchNormalization()(conv)
        conv = GatedLinearUnit()(conv)  # Apply GLU to residual connection
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    
    out = Activation('linear')(added)  # Maintain the linearity after residual addition

    # Repeat for additional depth
    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i + 1), activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = SpatialDropout1D(dropout)(block)
        
        block = GatedLinearUnit()(block)  # Apply GLU

        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i + 1), activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = SpatialDropout1D(dropout)(block)

        # Add residual connection
        added = Add()([block, out])
        out = Activation('linear')(added)  # Keep it linear for residual
        
    return out

#%% The proposed model, 
def ATCNet_with_CWT(n_classes, frequencies, in_chans=22, in_samples=1125, attention='mha',
                    eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                    tcn_depth=1, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                    tcn_activation='elu', fuse='average'):

    # Time-series Input
    input_eeg = Input(shape=(1, in_chans, in_samples))  # (batch_size, 1, channels, samples)
    
    # Frequency-series Input
    input_freq = Input(shape=(32, in_samples, in_chans))  # (batch_size, frequencies, samples, channels)

    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    from_logits = False

    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    # EEG Convolution Block for Time-Series Input
    input_eeg_permuted = Permute((3, 2, 1))(input_eeg)  # (batch_size, samples, channels, 1)
    block1 = TS_Conv_block(input_layer=input_eeg_permuted, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                         in_chans=in_chans, dropout=eegn_dropout)

    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)  # Squeeze sequence dimension if needed

    # Frequency Features for Attention Mask using TF_ConvModule
    freq_features = tf_conv_module(input_freq)  # Extract features from frequency input

    # Apply a 1D Convolution to the entire time series
    conv_layer = Conv1D(filters=tcn_filters, kernel_size=3, padding='same')(block1)  # Temporal convolution(SW complement)

    # Optionally, apply multi-head attention over the entire time series
    if attention == 'mha':
        conv_layer = attention_block(conv_layer, freq_features=freq_features, attention_model='mha')

    # Temporal Convolutional Network (TCN)
    tcn_output = TCN_block(input_layer=conv_layer, input_dimension=F2, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout)

    # Global Average Pooling to capture information over the entire sequence
    global_avg_pool = GlobalAveragePooling1D()(tcn_output)

    # Final Dense Layer for classification
    final_dense = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(global_avg_pool)

    # Final output layer (softmax or linear)
    if from_logits:
        out = Activation('linear', name='linear')(final_dense)
    else:
        out = Activation('softmax', name='softmax')(final_dense)

    # Create the model with EEG and CWT inputs
    return Model(inputs=[input_eeg, input_freq], outputs=out)
