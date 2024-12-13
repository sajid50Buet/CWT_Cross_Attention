def add_noise(data, noise_factor=0.001, seed=42):
    """Add Gaussian noise to the data."""
    return data + noise_factor * tf.random.normal(shape=tf.shape(data), dtype=tf.float32, seed=seed)

def scale_data(data, min_scale=0.9, max_scale=1.1, seed=42):
    """Scale the data by a random factor."""
    scale = tf.random.uniform([], minval=min_scale, maxval=max_scale, dtype=tf.float32, seed=seed)
    return data * scale

def apply_time_mask(data, max_mask_size=10, seed=42):
    """Apply a random time mask to the data."""
    time_steps = tf.shape(data)[1]
    time_mask_start = tf.random.uniform([], minval=0, maxval=time_steps, dtype=tf.int32, seed=seed)
    mask_size = tf.random.uniform([], minval=0, maxval=tf.minimum(max_mask_size, time_steps - time_mask_start), dtype=tf.int32, seed=seed)

    mask = tf.ones(shape=[tf.shape(data)[0], mask_size, tf.shape(data)[2]])
    mask = tf.pad(mask, [[0, 0], [time_mask_start, time_steps - time_mask_start - mask_size], [0, 0]], "CONSTANT")
    mask = tf.expand_dims(mask, axis=-1)

    return data * (1.0 - mask)

def mixup_augmentation(cwt_data, labels, alpha=0.2, seed=42):
    """Apply mixup augmentation to CWT data."""
    num_samples = tf.shape(cwt_data)[0]
    tf.random.set_seed(seed)

    # Choose random indices for mixup
    random_indices = tf.random.shuffle(tf.range(num_samples), seed=seed)
    mixup_lambda = tf.random.uniform([num_samples, 1, 1, 1], minval=0, maxval=alpha, dtype=tf.float32, seed=seed)

    # Ensure labels are broadcast-compatible and of type float32
    if len(labels.shape) == 2:  # Assuming [num_samples, num_classes]
        mixup_lambda_labels = tf.reshape(mixup_lambda, [num_samples, 1])

    # Convert labels to float32 if needed
    labels = tf.cast(labels, tf.float32)
    mixup_lambda_labels = tf.cast(mixup_lambda_labels, tf.float32)

    # Mix data and labels with chosen lambda
    mixed_data = mixup_lambda * cwt_data + (1 - mixup_lambda) * tf.gather(cwt_data, random_indices)
    mixed_labels = mixup_lambda_labels * labels + (1 - mixup_lambda_labels) * tf.gather(labels, random_indices)
    
    return mixed_data, mixed_labels

def augment_time_domain(timg, label, seed=42):
    """Apply multiple combinations of augmentations on time-domain data."""
    tf.random.set_seed(seed)

    # Apply all augmentations in batch
    noise_data = add_noise(timg, seed=seed)
    scale_data_only = scale_data(timg, seed=seed)
    noise_scale_data = scale_data(noise_data, seed=seed)
    noise_scale_mask_data = apply_time_mask(noise_scale_data, seed=seed)

    # Concatenate once for efficiency
    X_time_aug_combined = tf.concat([timg, noise_data, scale_data_only, noise_scale_data, noise_scale_mask_data], axis=0)
    y_time_aug_combined = tf.tile(label, [5, 1])

    return X_time_aug_combined, y_time_aug_combined

def shift_data(cwt_data, shift_range=(-3, 4), seed=42):
    """Shift the CWT data along the time dimension."""
    batch_size, freqs, time_steps, channels = tf.shape(cwt_data)
    shift = tf.random.uniform([], minval=shift_range[0], maxval=shift_range[1], dtype=tf.int32, seed=seed)

    if shift > 0:
        shifted = tf.concat([tf.zeros([batch_size, freqs, shift, channels]), cwt_data[:, :, :-shift, :]], axis=2)
    elif shift < 0:
        shifted = tf.concat([cwt_data[:, :, -shift:, :], tf.zeros([batch_size, freqs, -shift, channels])], axis=2)
    else:
        shifted = cwt_data

    return shifted

def augment_cwt(cwt_data, label, seed=42):
    """Apply multiple combinations of augmentations on CWT data."""
    tf.random.set_seed(seed)

    noise_cwt = add_noise(cwt_data, seed=seed)
    shift_cwt = shift_data(cwt_data, seed=seed)
    noise_shift_cwt = shift_data(noise_cwt, seed=seed)
    mixup_noise_shift_cwt, mixup_labels = mixup_augmentation(noise_shift_cwt, label, seed=seed)

    X_cwt_aug_combined = tf.concat([cwt_data, noise_cwt, shift_cwt, noise_shift_cwt, mixup_noise_shift_cwt], axis=0)
    y_cwt_aug_combined = tf.concat([label] * 4 + [mixup_labels], axis=0)

    return X_cwt_aug_combined, y_cwt_aug_combined

def time_and_cwt_augment(timg, cwt_data, label, seed=42):
    """Apply both time-domain and CWT augmentations."""
    X_time_aug_combined, y_time_aug_combined = augment_time_domain(timg, label, seed=seed)
    X_cwt_aug_combined, y_cwt_aug_combined = augment_cwt(cwt_data, label, seed=seed)

    return X_time_aug_combined, X_cwt_aug_combined, y_time_aug_combined
