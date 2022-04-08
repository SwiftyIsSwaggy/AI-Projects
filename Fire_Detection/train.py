import tensorflow as tf
import argparse
import os
import json
import numpy as np
import sklearn.metrics
import logging
import io
import itertools

from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D

IMG_SHAPE = 128

logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

HP_EPOCHS = hp.HParam("epochs", hp.IntInterval(1, 100))
HP_BATCH_SIZE = hp.HParam("batch-size", hp.Discrete([64, 128, 256, 512]))
HP_LR = hp.HParam("learning-rate", hp.RealInterval(0.0, 1.0))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["sgd", "adam", "rmsprop"]))

METRIC_ACCURACY = "accuracy"

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

def create_data_generators(root_dir, batch_sz):
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    image_gen_train = ImageDataGenerator(rescale = 1./255,
                    zoom_range = 0.5,
                    rotation_range=45,
                    horizontal_flip=True,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.2)
    train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_sz,
                                            directory=train_dir,
                                            shuffle=True,
                                            target_size=(IMG_SHAPE,IMG_SHAPE),
                                            class_mode='binary')
    image_gen_val = ImageDataGenerator(rescale=1./255)
    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_sz,
                            directory=val_dir,
                            target_size=(IMG_SHAPE,IMG_SHAPE),
                            class_mode='binary',
                            shuffle = False)
    print(train_data_gen.samples)
    print(train_data_gen.n)
    print(val_data_gen.samples)
    print(val_data_gen.n)
    #Find our size of datasets. each
    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plotImages(augmented_images)
    return train_data_gen, val_data_gen

def create_model(learning_rate, optimizer):
    model = Sequential()
    model.add(Conv2D(filters = 32,kernel_size = 3, padding = "same", input_shape = (IMG_SHAPE, IMG_SHAPE, 3), strides = 2))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 64, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 128, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 128, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 256, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 256, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))


    #Four round blocks, MobileNet has five instead
    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    #After 4 blocks
    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 1024, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))
    model.add(Conv2D(filters = 1024, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu6'))

    model.add(GlobalAveragePooling2D())

    model.add(Flatten())
    model.add(Dense(2, activation = 'softmax'))

    model.summary()
    if optimizer == "sgd":
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    else:
        raise Exception("Unknown optimizer", optimizer)

    model.compile(optimizer=opt,
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
    return model

def tb_plot_confusion_matrix(y_true, y_pred):
    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    classes_cnt = cm.shape[0]

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(classes_cnt)
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    cm_image = tf.expand_dims(image, 0)

    return cm_image

def main(args):
    # Initializing TensorFlow summary writer
    job_name = json.loads(os.environ.get("SM_TRAINING_ENV"))["job_name"]
    logs_dir = "{}/{}".format(args.tf_logs_path, job_name)
    logging.info("Writing TensorBoard logs to {}".format(logs_dir))
    tf_writer = tf.summary.create_file_writer(logs_dir)
    tf_writer.set_as_default()

    # Configuration of hyperparameters to visualize in TensorBoard
    hp.hparams_config(
        hparams=[HP_EPOCHS, HP_BATCH_SIZE, HP_LR, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
    )

    hparams = {
        HP_EPOCHS: args.epochs,
        HP_BATCH_SIZE: args.batch_size,
        HP_LR: args.learning_rate,
        HP_OPTIMIZER: args.optimizer,
    }

    local_output_dir = args.sm_model_dir
    local_root_dir = args.train
    batch_size = args.batch_size
    
    train_gen, val_gen = create_data_generators(local_root_dir, batch_size)

    model = create_model(args.learning_rate, args.optimizer)

    callbacks = []
    # TensorBoard callback to collect standard metrics, profiling informationg, and compute activation and weight histograms for the layers
    callbacks.append(
        TensorBoard(log_dir=logs_dir, update_freq="epoch", histogram_freq=1, profile_batch="5,35")
    )
    # TensorBoard logging hyperparameter
    callbacks.append(hp.KerasCallback(writer=logs_dir, hparams=hparams, trial_id=job_name))
    
    model.fit(
        train_gen,
        epochs=args.epochs,
        batch_size = args.batch_size,
        validation_data=val_gen,
        callbacks = callbacks
    )
    #Run this cell to export the model to s3 after being named by epochs
    model_name = "firenet"
    model_fullname = "{}_epochs_{}".format(model_name,args.epochs)
    model.save(os.path.join(local_output_dir,model_fullname, '1'))

    val_gen.reset()
    Y_pred = model.predict(val_gen, int(np.ceil(val_gen.n / float(batch_size))))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_gen.classes

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    logging.info("Test accuracy:{}".format(accuracy))

    # Calculating confusion matrix and logging it as an image for TensorBoard visualization.
    cm_image = tb_plot_confusion_matrix(y_true, y_pred)
    tf.summary.image("Confusion Matrix", cm_image, step=1)
    tf_writer.flush() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type = int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model-output', type=str, default = os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument('--train', type=str, default = os.environ.get('SM_CHANNEL_TRAINING'))

    parser.add_argument(
        "--tf-logs-path",
        type=str,
        required=True,
        help="Path used for writing TensorFlow logs. Can be S3 bucket.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="The number of steps to use for training."
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument(
        "--data-config", type=json.loads, default=os.environ.get("SM_INPUT_DATA_CONFIG")
    )
    parser.add_argument("--optimizer", type=str.lower, default="adam")
    
    args = parser.parse_args()
    main(args)
    
    
"""
#Confusion Matrix and Classification Report

print('Confusion Matrix')
print(confusion_matrix(val_data_gen.classes, y_pred))
print('Classification Report')
target_names = ['Fire', 'No Fire']
print(classification_report(val_data_gen.classes, y_pred, target_names=target_names))

cm = confusion_matrix(val_data_gen.classes, y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(val_data_gen.classes, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

prec, recall, _ = precision_recall_curve(val_data_gen.classes, y_pred)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
"""