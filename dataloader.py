# Main wrapper function for loading data, computing CWT, and selecting frequencies
def get_data(path, subject, frequencies, sampling_frequency, dataset='BCI2a', classes_labels='all', LOSO=False,
             isStandard=True, isShuffle=True, include_cwt=True, top_k=50, final_k=32):
    """
    Load the dataset, compute CWT, and automatically select the best frequencies.
    """
    # Load and split the dataset into training and testing
    if LOSO:
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        if dataset == 'BCI2a':
            X_train, y_train = load_BCI2a_data(path, subject, True)
            X_test, y_test = load_BCI2a_data(path, subject, False)

    # Shuffle the data if specified
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=seed)
        X_test, y_test = shuffle(X_test, y_test, random_state=seed)

    # Reshape training and testing data
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)

    N_te, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_te, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)

    # Standardize the data if specified
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    # Transpose and squeeze singleton dimensions
    X_train_transposed = tf.squeeze(tf.transpose(X_train, perm=[0, 2, 1, 3]), axis=2)
    X_test_transposed = tf.squeeze(tf.transpose(X_test, perm=[0, 2, 1, 3]), axis=2)

    # Step 1: Compute CWT for the data
    X_train_cwt = batch_cwt(tf.convert_to_tensor(X_train_transposed), frequencies, sampling_frequency)
    X_test_cwt = batch_cwt(tf.convert_to_tensor(X_test_transposed), frequencies, sampling_frequency)

    # Step 2: Select frequencies using Mutual Information + Random Forest
    y_train_labels = np.argmax(y_train_onehot, axis=1)
    selected_indices = select_best_frequencies(X_train_cwt.numpy(), y_train_labels, frequencies, top_k=top_k, final_k=final_k)

    # Fix indexing issue by converting indices into TensorFlow tensors
    selected_indices_tf = tf.convert_to_tensor(selected_indices, dtype=tf.int32)

    # Gather selected frequencies from the CWT tensors
    X_train_cwt_filtered = tf.gather(X_train_cwt, selected_indices_tf, axis=1)  # Gather along the frequency axis
    X_test_cwt_filtered = tf.gather(X_test_cwt, selected_indices_tf, axis=1)

    # Log useful information for debugging
    print(f"Selected frequencies: {frequencies[selected_indices]}")
    print(f"Filtered CWT shapes: Training={X_train_cwt_filtered.shape}, Testing={X_test_cwt_filtered.shape}")
    
    return (X_train, y_train, y_train_onehot, 
            X_test, y_test, y_test_onehot, 
            X_train_cwt_filtered, X_test_cwt_filtered)
