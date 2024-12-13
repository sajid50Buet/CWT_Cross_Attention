pip install tqdm
class CustomTQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        # Initialize the tqdm progress bar for epochs
        self.epochs_bar = tqdm(total=self.params['epochs'], position=0, desc='Epochs', unit='epoch')
        # Initialize variables to track best validation accuracy and loss, and their corresponding epochs
        self.best_val_acc = 0.0
        self.best_epoch_acc = 0
        self.best_val_loss = float('inf')  # Start with infinity for minimum comparison
        self.best_epoch_loss = 0

    def on_epoch_end(self, epoch, logs=None):
        # Retrieve relevant metrics from logs
        train_acc = logs.get('accuracy', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)
        val_loss = logs.get('val_loss', float('inf'))  # Default to infinity if not available

        # Update the best validation accuracy and epoch if the current accuracy is higher
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch_acc = epoch + 1  # Store the best epoch (1-based index)

        # Update the best validation loss and epoch if the current loss is lower
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch_loss = epoch + 1  # Store the best epoch (1-based index)

        # Update the progress bar description with metrics, including best validation accuracy and loss
        self.epochs_bar.set_description(
            f"Epoch {epoch + 1}, Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Valid Loss: {val_loss:.4f}, "
            f"Best Valid Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch_acc}), "
            f"Best Valid Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch_loss})"
        )

        # Move the progress bar by 1 epoch
        self.epochs_bar.update(1)

    def on_train_end(self, logs=None):
        # Close the tqdm progress bar at the end of training
        self.epochs_bar.close()

#%%
def draw_learning_curves(history, sub):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy - subject: ' + str(sub))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss - subject: ' + str(sub))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    # Generate confusion matrix plot
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + sub )
    plt.savefig(results_path + '/subject_' + sub + '.png')
    plt.show()


def draw_performance_barChart(num_sub, metric, label, mean_best_value):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub + 1))
    # Draw the bar chart for each subject's metric
    bars = ax.bar(x, metric, 0.5, label=label) 
    # Draw a dotted line for the overall mean of the best scores
    ax.axhline(y=mean_best_value, color='r', linestyle='--', label=f'Avg {label} ({mean_best_value:.4f})')
    # Add labels and titles
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title(f'Model {label} per Subject')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right')
    # Display the accuracy score above each bar
    for bar, score in zip(bars, metric):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate
            bar.get_height(),                  # Y-coordinate (top of the bar)
            f'{score:.4f}',                    # Text (formatted score)
            ha='center', va='bottom'           # Center text horizontally, below text vertically
        )  
    # Show the plot
    plt.show()
    
def plot_tsne(features, labels, title="t-SNE Plot", class_labels=['Left hand', 'Right hand', 'Foot', 'Tongue'],
              perplexity=40, learning_rate=100, n_pca_components=30, save_path='./tsne_plot.png'):
    # Reduce dimensions with PCA before applying t-SNE
#     pca = PCA(n_components=n_pca_components, random_state=42)
#     pca_results = pca.fit_transform(features)
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, init='pca', random_state=42)
#     tsne = TSNE(n_components=2, perplexity=perplexity,init='pca', random_state=166)
#     tsne_results = tsne.fit_transform(pca_results)
    tsne_results = tsne.fit_transform(features)
    # Normalize t-SNE results for consistent plotting
    x_min, x_max = tsne_results.min(0), tsne_results.max(0)
    tsne_normalized = (tsne_results - x_min) / (x_max - x_min)
    # Define specific colors for each class
    colors = ['red', 'blue', 'green', 'brown']
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels)):
        # Select data points belonging to the current class
        class_points = tsne_normalized[labels == label]
        plt.scatter(class_points[:, 0], class_points[:, 1], 
                    color=colors[i], label=class_labels[label], alpha=0.7)
    # Add plot details
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.xticks([])  # Remove x-axis ticks for cleaner visualization
    plt.yticks([])  # Remove y-axis ticks for cleaner visualization
    plt.legend()
    plt.show()
    
def train_and_test(dataset_conf, train_conf, results_path):
    # Remove the 'results' folder before training
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    in_exp = time.time()  # Start time for the overall experiment
    best_models = open(results_path + "/best_models.txt", "w")  # Log best models
    log_write = open(results_path + "/log.txt", "w")  # Log file

    # Dataset and training parameters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    include_cwt = dataset_conf.get('include_cwt')
    batch_size = train_conf.get('batch_size')
    model_name = train_conf.get('model')
    lr = train_conf.get('lr')
    epochs = train_conf.get('epochs')
    n_train = train_conf.get('n_train')
    from_logits = train_conf.get('from_logits')
    frequencies = dataset_conf.get('cwt_frequencies')
    sampling_frequency = dataset_conf.get('sampling_frequency')
    LearnCurves = train_conf.get('LearnCurves')
    classes_label = dataset_conf.get('cl_labels')
    patience = train_conf.get('patience')

    # Initialize arrays for storing training accuracy, kappa, test accuracy, kappa, and confusion matrices
    test_acc = np.zeros((n_sub, n_train))
    test_kappa = np.zeros((n_sub, n_train))
    cf_matrix = np.zeros([n_sub, n_train, n_classes, n_classes])
    test_precision = np.zeros((n_sub, n_train))
    test_recall = np.zeros((n_sub, n_train))
    test_f1 = np.zeros((n_sub, n_train))
    inference_time = 0
    
    # Ensure a consistent seed is set
    global_seed = 42
    tf.random.set_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Ensure deterministic operations
  
    for sub in tqdm(range(n_sub), desc="Training and testing subjects", unit="subject"):
        print(f'\nTraining on subject {sub + 1}')
        log_write.write(f'\nTraining on subject {sub + 1}\n')
        BestSubjAcc = 0
        bestTrainingHistory = []

        # Get training and test data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot, X_train_cwt, X_test_cwt = get_data(
            data_path, sub, frequencies, sampling_frequency, dataset=dataset, LOSO=LOSO, isStandard=isStandard, include_cwt=True
        )

        # Convert TensorFlow tensors to NumPy arrays if necessary
        X_train = X_train.numpy() if hasattr(X_train, 'numpy') else X_train
        y_train_onehot = y_train_onehot.numpy() if hasattr(y_train_onehot, 'numpy') else y_train_onehot
        X_train_cwt = X_train_cwt.numpy() if hasattr(X_train_cwt, 'numpy') else X_train_cwt

        # Augment the training data using your augmentation function
        X_train_aug, X_train_cwt_aug, y_train_aug = time_and_cwt_augment(X_train, X_train_cwt, y_train_onehot)

        # Convert to NumPy arrays if necessary
        X_train_aug = X_train_aug.numpy() if hasattr(X_train_aug, 'numpy') else X_train_aug
        y_train_aug = y_train_aug.numpy() if hasattr(y_train_aug, 'numpy') else y_train_aug
        X_train_cwt_aug = X_train_cwt_aug.numpy() if hasattr(X_train_cwt_aug, 'numpy') else X_train_cwt_aug
        # Print shapes after augmentation
        print(f"Augmented shapes:")
        print(f"X_train_aug shape: {X_train_aug.shape}")
        print(f"y_train_aug shape: {y_train_aug.shape}")
        print(f"X_train_cwt_aug shape: {X_train_cwt_aug.shape}")
    
        # Training loop with modifications
        for train in tqdm(range(n_train), desc=f"Training runs for subject {sub + 1}", unit="run"):
            # Control the seed for each training run
            #run_seed = train + global_seed
            run_seed = global_seed
            tf.random.set_seed(run_seed)
            np.random.seed(run_seed)
            random.seed(run_seed)
            
            in_run = time.time()
            filepath = os.path.join(results_path, 'saved_models', f'run-{train + 1}', f'subject-{sub + 1}.weights.h5')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Create the model and set initial weights
            model = getModel(model_name, dataset_conf, from_logits)
            initial_weights = model.get_weights()  # Save initial weights
            
            # Print model input configuration (only for first subject's first run)
            if sub == 0 and train == 0:
                print("Model input configuration:", model.inputs)
                model.summary()
                plot_model(model, to_file=os.path.join(results_path, 'model_summary.png'), show_shapes=True, show_layer_names=True)
            
            # Compile the model with gradient clipping
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits),
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # Make sure lr is defined properly
                metrics=['accuracy']
            )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience),
                CustomTQDMProgressBar()  # Custom progress bar
            ]
            
            # Load the initial weights before training each subject
            model.set_weights(initial_weights)
            
            # Train the model
            history = model.fit([X_train_aug, X_train_cwt_aug], y_train_aug,
                                validation_data=([X_test, X_test_cwt], y_test_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            # Evaluate the best model on validation data
            model.load_weights(filepath)

            # Evaluate the model on test data
            y_pred_test = model.predict([X_test, X_test_cwt])
            if from_logits:
                y_pred_test = tf.nn.softmax(y_pred_test).numpy().argmax(axis=-1)
            else:
                y_pred_test = y_pred_test.argmax(axis=-1)

            labels_test = y_test_onehot.argmax(axis=-1)
            test_acc[sub, train] = accuracy_score(labels_test, y_pred_test)
            test_kappa[sub, train] = cohen_kappa_score(labels_test, y_pred_test)
            test_precision[sub, train] = precision_score(labels_test, y_pred_test, average='weighted')
            test_recall[sub, train] = recall_score(labels_test, y_pred_test, average='weighted')
            test_f1[sub, train] = f1_score(labels_test, y_pred_test, average='weighted')
            cf_matrix[sub, train, :, :] = confusion_matrix(labels_test, y_pred_test, normalize='true')

            out_run = time.time()

            # Log the performance after each run
            info = f'Subject: {sub + 1}   seed {train + 1}   time: {(out_run - in_run) / 60:.1f} m   '
            info += f'test_acc: {test_acc[sub, train]:.4f}  test_kappa: {test_kappa[sub, train]:.4f}  '
            info += f'test_precision: {test_precision[sub, train]:.4f}  test_recall: {test_recall[sub, train]:.4f}  test_f1: {test_f1[sub, train]:.4f}'
            print(info)
            log_write.write(info + '\n')
            
            # Save best run for the subject
            if BestSubjAcc < test_acc[sub, train]:
                BestSubjAcc = test_acc[sub, train]
                bestTrainingHistory = history

            # Clear GPU and RAM after each run
            del history  # Remove large objects to free memory
            K.clear_session()  # Clear TensorFlow session
            gc.collect()  # Collect garbage to release memory
            
            # If GPU memory is utilized, reset GPU memory
            if tf.config.experimental.get_visible_devices('GPU'):
                tf.keras.backend.clear_session()  # Clear GPU memory
                try:
                    tf.config.experimental.set_memory_growth(tf.config.experimental.get_visible_devices('GPU')[0], True)
                except:
                    pass

        # Test the best model after all runs
        runs = os.listdir(results_path + "/saved_models")
        X_test, X_test_cwt = np.array(X_test), np.array(X_test_cwt)
        n_samples = X_test.shape[0]

        for seed in range(len(runs)):
            model.load_weights(f'{results_path}/saved_models/run-{seed + 1}/subject-{sub + 1}.weights.h5')
            y_pred_test = []
            start_time = time.time()

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                y_pred_batch = model.predict([X_test[i:batch_end], X_test_cwt[i:batch_end]]).argmax(axis=-1)
                y_pred_test.extend(y_pred_batch)

            inference_time += (time.time() - start_time) / n_samples

            y_pred_test = np.array(y_pred_test)
            labels_test = y_test_onehot.argmax(axis=-1)

            # Compute metrics
            precision = precision_score(labels_test, y_pred_test, average='weighted')
            recall = recall_score(labels_test, y_pred_test, average='weighted')
            f1 = f1_score(labels_test, y_pred_test, average='weighted')

            test_precision[sub, seed] = precision
            test_recall[sub, seed] = recall
            test_f1[sub, seed] = f1
            test_acc[sub, seed] = accuracy_score(labels_test, y_pred_test)
            test_kappa[sub, seed] = cohen_kappa_score(labels_test, y_pred_test)
            cf_matrix[sub, seed, :, :] = confusion_matrix(labels_test, y_pred_test, normalize='true')
            
            # Clean up after each seed-based evaluation
            gc.collect()
            K.clear_session()

        # Logging and plotting results
        best_run = np.argmax(test_acc[sub, :])  # Best run based on accuracy
        best_models.write(f'subject-{sub + 1}.weights.h5 for run-{best_run + 1}\n')
        
        # Load the best model weights for the best run
        best_model_path = f'{results_path}/saved_models/run-{best_run + 1}/subject-{sub + 1}.weights.h5'
        model.load_weights(best_model_path)

        # Plot learning curves if required
        if LearnCurves:
            print('Plotting Learning Curves .......')
            draw_learning_curves(bestTrainingHistory, sub + 1)
            
        draw_confusion_matrix(cf_matrix[sub, best_run, :, :], f'subject_{sub + 1}', results_path, classes_label)
        print(f'Confusion matrix plotted for subject {sub + 1}.')
        
        # Define the feature extraction model
        feature_model = Model(inputs=[model.input[0], model.input[1]], outputs=model.get_layer('global_average_pooling1d').output)

        # Get learned features for t-SNE
        learned_features = feature_model.predict([X_test, X_test_cwt])
        labels_test = y_test_onehot.argmax(axis=-1)  # True class labels

        # Plot t-SNE using learned features
        print(f"Plotting t-SNE for learned features...")
        plot_tsne(learned_features, labels_test)
        
#        # Assuming 'model' is your trained model and you have a layer named 'ts__attention_module_1'
#         attention_layer_name = 'ts__attention_module_1'
#         attention_extractor = Model(inputs=model.input, 
#                                     outputs=model.get_layer(attention_layer_name).output)

#         # Generate Attention Maps for a Batch of Test Data
#         # Replace `attention_inputs` with the actual test inputs
#         attention_maps = attention_extractor.predict([X_test, X_test_cwt])

#         # Visualize Attention Maps for the First Sample in the Batch
#         # Check the shape of attention_maps to understand its dimensions
#         sample_attention_map = attention_maps[0]  # Select the first sample in the batch

#         # Plotting the attention heatmap
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(sample_attention_map, cmap='viridis')
#         plt.title(f'Attention Map for Layer {attention_layer_name}')
#         plt.xlabel('Key (input time steps/channels)')
#         plt.ylabel('Query (output time steps/channels)')
#         plt.show()

        # After each subject
        del X_train, y_train_onehot, X_train_cwt
        del y_pred_test, labels_test, X_test, X_test_cwt, y_test_onehot
        del X_train_aug, X_train_cwt_aug, y_train_aug
        del model, bestTrainingHistory
        gc.collect()
        K.clear_session()
        
    # Timing the end of the experiment
    out_exp = time.time()
    # Prepare header for testing performance logging
    head1_test = head2_test = '                '
    for sub in range(n_sub): 
        head1_test += f'sub_{sub + 1}   '
        head2_test += '-----   '
    head1_test += ' average'
    head2_test += ' -------'
    
    # Prepare test performance logging with additional metrics
    test_info = f'\n---------------------------------\nTest performance (acc %, kappa, precision, recall, f1):\n---------------------------------\n{head1_test}\n{head2_test}'
    
    # Print test performance for each seed and subject-wise
    for run in range(n_train):  # Use n_train to ensure it matches the training runs
        test_info += f'\nSeed {run + 1}:'
        test_info_acc = '(acc %)  '
        test_info_k = '   (k-sco)      '
        test_info_prec = '   (prec)       '
        test_info_recall = '  (recall)      '
        test_info_f1 = '    (f1)        '  
        for sub in range(n_sub): 
            test_info_acc += f'{test_acc[sub, run] * 100:.2f}    '
            test_info_k += f'{test_kappa[sub, run]:.3f}   '
            test_info_prec += f'{test_precision[sub, run]:.3f}   '
            test_info_recall += f'{test_recall[sub, run]:.3f}   '
            test_info_f1 += f'{test_f1[sub, run]:.3f}   '
        test_info_acc += f' {np.average(test_acc[:, run]) * 100:.2f}   '
        test_info_k += f'  {np.average(test_kappa[:, run]):.3f}   '
        test_info_prec += f'  {np.average(test_precision[:, run]):.3f}   '
        test_info_recall += f'  {np.average(test_recall[:, run]):.3f}   '
        test_info_f1 += f'  {np.average(test_f1[:, run]):.3f}   '   
        test_info += test_info_acc + '\n' + test_info_k + '\n' + test_info_prec + '\n' + test_info_recall + '\n' + test_info_f1

    # Subject-wise averages across all seeds
    test_info += f'\n\nSubject-wise averages across all seeds:\n'
    test_info += ' (acc %)        '
    test_info_kappa = '  (k-sco)       '
    test_info_prec_avg = '  (prec)        '
    test_info_recall_avg = '  (recall)      '
    test_info_f1_avg = '    (f1)        '

    subject_best_acc_list = []
    subject_best_kappa_list = []
    subject_best_prec_list = []
    subject_best_recall_list = []
    subject_best_f1_list = []

    for sub in range(n_sub): 
        # Calculate averages
        subject_avg_acc = np.average(test_acc[sub, :])
        subject_avg_kappa = np.average(test_kappa[sub, :])
        subject_avg_prec = np.average(test_precision[sub, :])
        subject_avg_recall = np.average(test_recall[sub, :])
        subject_avg_f1 = np.average(test_f1[sub, :])

        test_info += f'{subject_avg_acc * 100:.2f}    '
        test_info_kappa += f'{subject_avg_kappa:.3f}   '
        test_info_prec_avg += f'{subject_avg_prec:.3f}   '
        test_info_recall_avg += f'{subject_avg_recall:.3f}   '
        test_info_f1_avg += f'{subject_avg_f1:.3f}   '

        # Calculate best values for each subject
        subject_best_acc_list.append(np.max(test_acc[sub, :]))
        subject_best_kappa_list.append(np.max(test_kappa[sub, :]))
        subject_best_prec_list.append(np.max(test_precision[sub, :]))
        subject_best_recall_list.append(np.max(test_recall[sub, :]))
        subject_best_f1_list.append(np.max(test_f1[sub, :]))

    # Overall averages
    test_info += f' {np.average(test_acc) * 100:.2f}   '
    test_info_kappa += f'  {np.average(test_kappa):.3f}   '
    test_info_prec_avg += f'  {np.average(test_precision):.3f}   '
    test_info_recall_avg += f'  {np.average(test_recall):.3f}   '
    test_info_f1_avg += f'  {np.average(test_f1):.3f}   '
    test_info += '\n' + test_info_kappa + '\n' + test_info_prec_avg + '\n' + test_info_recall_avg + '\n' + test_info_f1_avg

    # Display subject-wise best results
    test_info += f'\n\nSubject-wise best results across all runs:\n'
    test_info += '  (best acc %)  '
    test_info_kappa_best = '  (best k-sco)  '
    test_info_prec_best = '  (best prec)   '
    test_info_recall_best = '  (best recall) '
    test_info_f1_best = '    (best f1)   '

    for sub in range(n_sub):
        test_info += f'{subject_best_acc_list[sub] * 100:.2f}    '
        test_info_kappa_best += f'{subject_best_kappa_list[sub]:.3f}   '
        test_info_prec_best += f'{subject_best_prec_list[sub]:.3f}   '
        test_info_recall_best += f'{subject_best_recall_list[sub]:.3f}   '
        test_info_f1_best += f'{subject_best_f1_list[sub]:.3f}   '

    # Overall means of best scores
    mean_best_acc = np.mean(subject_best_acc_list)
    mean_best_kappa = np.mean(subject_best_kappa_list)
    mean_best_prec = np.mean(subject_best_prec_list)
    mean_best_recall = np.mean(subject_best_recall_list)
    mean_best_f1 = np.mean(subject_best_f1_list)

    test_info += f'  {mean_best_acc * 100:.2f}   '
    test_info_kappa_best += f'  {mean_best_kappa:.3f}   '
    test_info_prec_best += f'  {mean_best_prec:.3f}   '
    test_info_recall_best += f'  {mean_best_recall:.3f}   '
    test_info_f1_best += f'  {mean_best_f1:.3f}   '
    test_info += '\n' + test_info_kappa_best + '\n' + test_info_prec_best + '\n' + test_info_recall_best + '\n' + test_info_f1_best

    # Overall averages for testing performance
    test_info += f'\n----------------------------------\nAverage - all seeds (acc %): {np.average(test_acc) * 100:.2f}\n'
    test_info += f'                    (k-sco): {np.average(test_kappa):.3f}\n'
    test_info += f'                    (prec):  {np.average(test_precision):.3f}\n'
    test_info += f'                    (recall):{np.average(test_recall):.3f}\n'
    test_info += f'                    (f1):    {np.average(test_f1):.3f}\n'
    test_info += f'\nSubject-wise best average (acc %): {mean_best_acc*100:.2f}\n'
    test_info += f'                          (k-sco): {mean_best_kappa:.3f}\n'
    test_info += f'                          (prec):  {mean_best_prec:.3f}\n'
    test_info += f'                          (recall):{mean_best_recall:.3f}\n'
    test_info += f'                           (f1):   {mean_best_f1:.3f}\n'
    test_info += f'\nInference time: {inference_time / len(runs):.2f} ms per trial\n'
    test_info += '----------------------------------\n'

    # Final output
    info = test_info

    # Print final results and write to log file
    print(info)
    log_write.write(info + '\n')

    # Save confusion matrices and inference time
    np.save(os.path.join(results_path, 'confusion_matrix.npy'), cf_matrix)
    np.save(os.path.join(results_path, 'inference_time.npy'), inference_time)
    np.save(os.path.join(results_path, 'precision.npy'), test_precision)
    np.save(os.path.join(results_path, 'recall.npy'), test_recall)
    np.save(os.path.join(results_path, 'f1_score.npy'), test_f1)
    
    draw_performance_barChart(n_sub, subject_best_acc_list, 'Testing Accuracy', mean_best_acc)
    draw_performance_barChart(n_sub, subject_best_kappa_list, 'Testing Kappa Score', mean_best_kappa)
    draw_performance_barChart(n_sub, subject_best_prec_list, 'Testing Precision', mean_best_prec)
    draw_performance_barChart(n_sub, subject_best_recall_list, 'Testing Recall', mean_best_recall)
    draw_performance_barChart(n_sub, subject_best_f1_list, 'Testing F1-Score', mean_best_f1)
    draw_confusion_matrix(cf_matrix.mean((0, 1)), 'All', results_path, classes_labels)
    
    # Close log files    
    best_models.close()   
    log_write.close()

    # Total experiment time
    print(f'\nTotal Experiment Time: {(time.time() - in_exp) / 60:.1f} minutes.')
    
#%%
def getModel(model_name, dataset_conf, from_logits = False):
    
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')
    frequencies = dataset_conf.get('cwt_frequencies') 

    # Select the model
    if(model_name == 'ATCNet_with_CWT'):
        # Train using the proposed ATCNet model: https://ieeexplore.ieee.org/document/9852687
        model = ATCNet_with_CWT( 
            # Dataset parameters
            n_classes = n_classes, 
            in_chans = n_channels, 
            in_samples = in_samples,
            frequencies= frequencies,
            # Sliding window (SW) parameter
            #n_windows = 5, 
            # Attention (AT) block parameter
            attention = 'mha', # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1 = 16,
            eegn_D = 2, 
            eegn_kernelSize = 64,
            eegn_poolSize = 7,
            eegn_dropout = 0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 32,
            tcn_dropout = 0.3, 
            tcn_activation='elu',
            )     
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model


in_samples = 1000
n_channels = 22
n_sub = 9
n_classes = 4
classes_labels = ['Left hand', 'Right hand','Foot','Tongue']
data_path ='/kaggle/input/bcic-iv-2amatlab-version/'
dataset = 'BCI2a'
lr = 0.0009

# Create a folder to store the results of the experiment
results_path = os.getcwd() + "/results"
if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory if it does not exist 
    
    # Set dataset paramters 
# Set dataset parameters
dataset_conf = {
    'name': dataset,
    'n_classes': n_classes,
    'cl_labels': classes_labels,
    'n_sub': n_sub,
    'n_channels': n_channels,
    'in_samples': in_samples,
    'data_path': data_path,
    'cwt_frequencies': np.linspace(1,100,100),#frequencies,  # CWT frequency range from 0.5 to 100 Hz 
    'isStandard': True,
    'LOSO': False,
    'include_cwt': True,
    'sampling_frequency': 250  # Raw EEG signal sampling frequency
}

train_conf = { 'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': lr,'n_train': 1,
                  'LearnCurves': True, 'from_logits': False, 'model':'ATCNet_with_CWT'}
    
results_path= '/kaggle/working/results'
# Call the training function
train_and_test(dataset_conf,train_conf, results_path)
