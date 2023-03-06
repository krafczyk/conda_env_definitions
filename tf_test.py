import tensorflow as tf
import tensorflow_datasets as tfds

train_ds, test_ds = tfds.load(
    'mnist',
    split = ['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=False)

def normalize_data(x, y):
    return tf.cast(x, tf.float32)/255., y

mdl = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

mdl.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam())

mdl.fit(train_ds.map(normalize_data).batch(32), epochs=5)

print("Training finished.")
