import tensorflow as tf
import argparse
import os
import json

def create_model():
    model = tf.keras.Sequential([
        tf.keras.applications.mobilenet_v2.MobileNetV2(include_top = False,
                                                      pooling = 'average',
                                                      input_shape = (128, 128, 3)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.layers[0].trainable = False
    model.compile(
        loss = 'binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model
def create_data_generators(root_dir, batch_sz):
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
        horizontal_flip = True,
        zoom_range=[0.8,1.2],
        rotation_range = 20
    ).flow_from_directory(
        os.path.join(root_dir, 'train'),
        target_size=(128,128),
        batch_size = batch_sz,
        class_mode='binary'
    )
    val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,

    ).flow_from_directory(
        os.path.join(root_dir, 'validation'),
        target_size=(128,128),
        batch_size = batch_sz,
        class_mode='binary'
    )
    return train_data_generator, val_data_generator
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type = int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=int(np.ceil(train_gen.n / float(16)))
    parser.add_argument('--val_steps', type=int, default=int(np.ceil(val_gen.n / float(16)))
    
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm_model_dir', type=str, default = os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument('--train', type=str, default = os.environ.get('SM_CHANNEL_TRAINING'))
    
    args, _ = parser.parse_known_args()
    local_output_dir = args.sm_model_dir
    local_root_dir = args.train
    batch_size = args.batch_size
                        
    train_gen, val_gen = create_data_generators(local_root_dir, batch_size)
    
    model = create_model()
    
    _ = model.fit(
        x=train_gen,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_data=val_gen,
        validation_steps=args.val_steps
    )
    
    model.save(os.path.join(local_output_dir, 'coursera_proj_model', '1'))
