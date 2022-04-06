import tensorflow as tf
import numpy as np
import os
import numpy as np 
import matplotlib.pyplot as plt
import glob
import shutil
from tensorboard.plugins.hparams import api as hp

#---
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, DepthwiseConv2D, AveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,auc, roc_auc_score,precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay,RocCurveDisplay,ConfusionMatrixDisplay

job_name = json.loads(os.environ.get("SM_TRAINING_ENV"))["job_name"]
        logs_dir = "{}/{}".format(args.tf_logs_path, job_name)
    logging.info("Writing TensorBoard logs to {}".format(logs_dir))
    tf_writer = tf.summary.create_file_writer(logs_dir)
    tf_writer.set_as_default()

#---
_URL = 'https://fire-net-datasets.s3.amazonaws.com/Training_Dataset.zip'

zip_file = tf.keras.utils.get_file(origin=_URL,extract=True)  
#This will ge the file and extract it to a directory and extract to /Training Dataset

#---
print(os.path.dirname(zip_file))
#This function returns the directory of the extracted folder without the extracted folder inclusive

#---
base_dir = os.path.join(os.path.dirname(zip_file), 'Training Dataset')
#A good way to add the directory of the extracted folder and also the extracted folder itself.
print(base_dir)

#---
LOG_DIR = os.path.join(os.getcwd(), "Fire_Detection/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


#---
classes = ['Fire', 'NoFire']

#---
for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  train, val = images[:round(len(images)*0.7)], images[round(len(images)*0.7):]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))

#---
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

#---
batch_size = 100
IMG_SHAPE = 128

#---
image_gen_train = ImageDataGenerator(rescale = 1./255,
                    zoom_range = 0.5,
                    rotation_range=45,
                    horizontal_flip=True,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.2)
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                            directory=train_dir,
                                            shuffle=True,
                                            target_size=(IMG_SHAPE,IMG_SHAPE),
                                            class_mode='binary')
print(train_data_gen.samples)
print(train_data_gen.n)
train_data_num = train_data_gen.samples
#Find our size of datasets. each

#---
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

#---
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#---
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                            directory=val_dir,
                            target_size=(IMG_SHAPE,IMG_SHAPE),
                            class_mode='binary',
                            shuffle = False)

#---
print(val_data_gen.samples)
print(val_data_gen.n)
val_data_num = val_data_gen.samples

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

#Hyperparamters to visualize in tensorboard
HP_EPOCHS = hp.HParam("epochs", hp.IntInterval(1, 150))
HP_BATCH_SIZE = hp.HParam("batch-size", hp.Discrete([64, 128, 256, 512]))
HP_LR = hp.HParam("learning-rate", hp.RealInterval(0.0, 1.0))

METRIC_ACCURACY = "accuracy"

#---
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



#---
EPOCHS = 100
model.compile(optimizer="adam",
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

history = model.fit(train_data_gen,epochs= EPOCHS,
                steps_per_epoch = int(np.ceil(train_data_gen.n / float(batch_size))),
                validation_data = val_data_gen,
                validation_steps = int(np.ceil(val_data_gen.n / float(batch_size))))

#---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


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


