import tensorflow as tf

class DataGenerator:
    def __init__(self):
        pass
    
    @tf.function
    def read_image(self, filename):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [64, 64])
        return image
        
    @tf.function
    def parse_function(self, filename, label):
        image = self.read_image(filename)
        return image, label
    
    @tf.function
    def parse_function_inference(self, filename):
        image = self.read_image(filename)
        return image

    @tf.function
    def train_preprocess(self, image, label):
        seed = 42
        # some random augmentations
#         image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        # Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def get_dataset(self, filenames, labels, batch_size=5, n_prefetch=1, training=True):
        # prepare dataset
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.map(self.parse_function, num_parallel_calls=4)
        if training:
            dataset = dataset.map(self.train_preprocess, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(n_prefetch)
        return dataset
    
    def get_dataset_inference(self, filenames, batch_size=5, n_prefetch=1):
        # prepare dataset
        dataset = tf.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.map(self.parse_function_inference, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(n_prefetch)
        return dataset
