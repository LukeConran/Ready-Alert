import tensorflow as tf

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("Built with CUDA: ", tf.test.is_built_with_cuda())

# Test GPU usage
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU")
    # Optional: limit GPU memory growth to avoid allocation issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("TensorFlow is using CPU only")