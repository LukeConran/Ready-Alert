import tensorflow as tf
import time
import numpy as np

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Create a simple model for testing
def create_test_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# Generate test data
batch_size = 64
X_train = np.random.random((1000, 784)).astype(np.float32)
y_train = np.random.randint(0, 10, (1000,))

print(f"\nTesting with batch size: {batch_size}")
print("Data shape:", X_train.shape)

# Test CPU performance
print("\n=== CPU Test ===")
with tf.device('/CPU:0'):
    cpu_model = create_test_model()
    start_time = time.time()
    cpu_model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=0)
    cpu_time = time.time() - start_time
    print(f"CPU training time: {cpu_time:.2f} seconds")

# Test GPU performance (if available)
if tf.config.list_physical_devices('GPU'):
    print("\n=== GPU Test ===")
    with tf.device('/GPU:0'):
        gpu_model = create_test_model()
        start_time = time.time()
        gpu_model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=0)
        gpu_time = time.time() - start_time
        print(f"GPU training time: {gpu_time:.2f} seconds")
        
        if cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\nSpeedup: {speedup:.2f}x faster on GPU")
else:
    print("\nNo GPU available for comparison")

# Monitor GPU usage during training
print("\n=== Real-time GPU Usage Test ===")
if tf.config.list_physical_devices('GPU'):
    # Larger model and data for better GPU utilization
    large_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    large_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Larger dataset
    X_large = np.random.random((5000, 784)).astype(np.float32)
    y_large = np.random.randint(0, 10, (5000,))
    
    print("Training larger model - check Task Manager or nvidia-smi to see GPU usage...")
    start_time = time.time()
    large_model.fit(X_large, y_large, batch_size=128, epochs=10, verbose=1)
    total_time = time.time() - start_time
    print(f"Large model training time: {total_time:.2f} seconds")

print("\nTip: Run 'nvidia-smi' in another terminal during training to monitor GPU usage!")