import tensorflow as tf
import numpy as np
import os
import numpy as np 
import matplotlib.pyplot as plt
import glob
import shutil
from tensorboard.plugins.hparams import api as hp
import logging
import json
import argparse

#---
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Activation, BatchNormalization, DepthwiseConv2D, AveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,auc, roc_auc_score,precision_recall_curve, accuracy_score
from sklearn.metrics import PrecisionRecallDisplay,ConfusionMatrixDisplay
#Hyperparamters to visualize in tensorboard
HP_EPOCHS = hp.HParam("epochs", hp.IntInterval(1, 150))
HP_BATCH_SIZE = hp.HParam("batch-size", hp.Discrete([64, 128, 256, 512]))
HP_LR = hp.HParam("learning-rate", hp.RealInterval(0.0, 1.0))

METRIC_ACCURACY = "accuracy"

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

    # Importing datasets
    
    model = Sequential()

    model.add(Conv2D(filters = 32,kernel_size = 3, padding = "same", input_shape = (IMG_SHAPE, IMG_SHAPE, 3)), strides = 2)
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 128, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 128, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 256, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 256, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    #Four round blocks, MobileNet has five instead
    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 512, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #After 4 blocks
    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 1024, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D(kernel_size = 3, strides =2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 1024, kernel_size = 1, padding = "same", strides = 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(AveragePooling2D(7,7), strides=(1,1))

    model.add(Flatten())
    model.add(Dense(2, activation = 'softmax'))

#--- [markdown]
# ### Places to Test
# 1. Overfit first so that you see maximum number of epoch. (Overfits above 60)
# 2. After overfitting, find the epoch where you want to reduce the learning rate. (This is 60 epoch)
# 3. Test with discarding dropout and use batch normalization. (Performance Accuracy) **Test this
# 4. Add a Dense layer of 128. (Accuracy) //No need, parameters already high.
# 5. MaxPooling to default stride (Accuracy) //Will increase computation.
# 6. After finding best epoch, use reduce early and stop. (Generalization)// No need, found the optimum range at about 60 epoch.
# 7. Convert to depthwise separable convolution and increase layers.

#---
    model.summary()

#Mobile Net had 2 million parameters



    model.compile(optimizer="adam",
                    loss=tf.keras.losses.sparse_categorical_crossentropy,
                    metrics=['accuracy'])
    callbacks = []
    # TensorBoard callback to collect standard metrics, profiling informationg, and compute activation and weight histograms for the layers
    callbacks.append(
        TensorBoard(log_dir=logs_dir, update_freq="epoch", histogram_freq=1, profile_batch="5,35")
    )
    # TensorBoard logging hyperparameter
    callbacks.append(hp.KerasCallback(writer=logs_dir, hparams=hparams, trial_id=job_name))

    # Train the model
    model.fit(
        train_data_gen,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=validation_data_gen,
        callbacks=callbacks,
    )

    # Saving trained model
    model.save(args.model_output + "/1")

    # Converting validation dataset to numpy array
    validation_array = np.array(list(validation_data_gen.unbatch().take(-1).as_numpy_iterator()))
    test_x = np.stack(validation_array[:, 0])
    test_y = np.stack(validation_array[:, 1])

    # Use the model to predict the labels
    test_predictions = model.predict(test_x)
    test_y_pred = np.argmax(test_predictions, axis=1)
    test_y_true = np.argmax(test_y, axis=1)

    # Evaluating model accuracy and logging it as a scalar for TensorBoard hyperparameter visualization.
    accuracy = accuracy_score(test_y_true, test_y_pred)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    logging.info("Test accuracy:{}".format(accuracy))

    # Calculating confusion matrix and logging it as an image for TensorBoard visualization.
    cm_image = tb_plot_confusion_matrix(test_y_true, test_y_pred)
    tf.summary.image("Confusion Matrix", cm_image, step=1)
    tf_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        help="The directory where the CIFAR-10 input data is stored.",
    )
    parser.add_argument(
        "--validation",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_VALIDATION"),
        help="The directory where the CIFAR-10 input data is stored.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
        help="The directory where the trained model will be stored.",
    )
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
    parser.add_argument("--model_dir", type=str)

    args = parser.parse_args()
    main(args)

def tb_plot_confusion_matrix(y_true, y_pred):
    # Calculate the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)

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

#---


#---
#Run this on AWS 
EFS_PATH_LOG_DIR = "/".join(LOG_DIR.strip("/").split('/')[1:-1])
print (EFS_PATH_LOG_DIR)


"""
Run this command in the terminal

pip install tensorboard
tensorboard --logdir <EFS_PATH_LOG_DIR>

To launch TensorBoard, copy your Studio URL and replace lab? with proxy/6006/ as follows. You must include the trailing / character.

https://<YOUR_URL>.studio.region.sagemaker.aws/jupyter/default/proxy/6006/

"""

#---
#Run this on Colab or other places

#Can run this before training to view the logs before training occurs or run after training to view after training

#---
#Confusion Matrix and Classification Report
val_data_gen.reset()
Y_pred = model.predict(val_data_gen, int(np.ceil(val_data_gen.n / float(batch_size))))
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(val_data_gen.classes, y_pred))
print('Classification Report')
target_names = ['Fire', 'No Fire']
print(classification_report(val_data_gen.classes, y_pred, target_names=target_names))

#---
#Confusion Matrix and Classification Report

#Something is Wrong
val_data_gen.reset()
Y_pred = model.predict(val_data_gen, int(np.ceil(val_data_gen.n / float(batch_size))))
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(val_data_gen.classes, y_pred))
print('Classification Report')
target_names = ['Fire', 'No Fire']
print(classification_report(val_data_gen.classes, y_pred, target_names=target_names))

#---
cm = confusion_matrix(val_data_gen.classes, y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()

#---
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

#---
prec, recall, _ = precision_recall_curve(val_data_gen.classes, y_pred)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

#---
model_name = "firenet"
model_fullname = "{}_epochs_{}".format(model_name,EPOCHS)

#---
export_path_keras = "./{}.h5".format(model_fullname)
print(export_path_keras)

model.save(export_path_keras)

#---
export_path_sm = "./{}".format(model_fullname)
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

#---
!ls {export_path_sm}

#---
!zip -r {model_fullname}.zip {export_path_sm}

#---
#Download saved model to Disk
try:
  from google.colab import files
  files.download('./{}.zip'.format(model_fullname)
except ImportError:
  pass


