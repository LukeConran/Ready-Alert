import tensorflow as tf
import time
import numpy as np
import os

print("=== TensorFlow GPU Diagnostic ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA built: {tf.test.is_built_with_cuda()}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Print CUDA/cuDNN versions
build_info = tf.sysconfig.get_build_info()
print(f"CUDA version (built with): {build_info.get('cuda_version', 'Unknown')}")
print(f"cuDNN version (built with): {build_info.get('cudnn_version', 'Unknown')}")

# Check environment variables
print(f"\nCUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"PATH contains CUDA: {'CUDA' in os.environ.get('PATH', '')}")

# Check if operations actually run on GPU
print("\n=== Testing Operation Placement ===")
if tf.config.list_physical_devices('GPU'):
    # Force GPU
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        
        # Time a simple operation
        start = time.time()
        c = tf.matmul(a, b)
        gpu_op_time = time.time() - start
        print(f"GPU matrix multiply device: {c.device}")
        print(f"GPU operation time: {gpu_op_time:.4f}s")
    
    # Force CPU for comparison
    with tf.device('/CPU:0'):
        a_cpu = tf.random.normal([1000, 1000])
        b_cpu = tf.random.normal([1000, 1000])
        
        start = time.time()
        c_cpu = tf.matmul(a_cpu, b_cpu)
        cpu_op_time = time.time() - start
        print(f"CPU matrix multiply device: {c_cpu.device}")
        print(f"CPU operation time: {cpu_op_time:.4f}s")
    
    print(f"GPU vs CPU ratio: {gpu_op_time/cpu_op_time:.2f} (should be < 1.0)")

# Test memory allocation
print("\n=== GPU Memory Test ===")
if tf.config.list_physical_devices('GPU'):
    try:
        # Try to allocate GPU memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            
        # Allocate progressively larger tensors
        for size in [100, 1000, 5000]:
            with tf.device('/GPU:0'):
                x = tf.random.normal([size, size])
                print(f"Successfully allocated {size}x{size} tensor on GPU")
                del x  # Free memory
                
    except Exception as e:
        print(f"GPU memory allocation failed: {e}")

# Test actual training with detailed timing
print("\n=== Detailed Training Test ===")

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Generate data
X = np.random.random((2000, 784)).astype(np.float32)
y = np.random.randint(0, 10, (2000,))

# Time data preparation
print("Data preparation complete")

# CPU training with timing breakdown
print("\n--- CPU Training ---")
with tf.device('/CPU:0'):
    cpu_model = create_model()
    cpu_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Time compilation
    start = time.time()
    _ = cpu_model(X[:1])  # Force compilation
    compile_time = time.time() - start
    print(f"Model compilation time (CPU): {compile_time:.4f}s")
    
    # Time actual training
    start = time.time()
    cpu_history = cpu_model.fit(X, y, batch_size=64, epochs=3, verbose=0)
    cpu_train_time = time.time() - start
    print(f"CPU training time: {cpu_train_time:.4f}s")

# GPU training with timing breakdown
if tf.config.list_physical_devices('GPU'):
    print("\n--- GPU Training ---")
    with tf.device('/GPU:0'):
        gpu_model = create_model()
        gpu_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Time compilation
        start = time.time()
        _ = gpu_model(X[:1])  # Force compilation
        gpu_compile_time = time.time() - start
        print(f"Model compilation time (GPU): {gpu_compile_time:.4f}s")
        
        # Time data transfer + training
        start = time.time()
        gpu_history = gpu_model.fit(X, y, batch_size=64, epochs=3, verbose=0)
        gpu_train_time = time.time() - start
        print(f"GPU training time: {gpu_train_time:.4f}s")
        
        print(f"\nSpeedup: {cpu_train_time/gpu_train_time:.2f}x")
        if gpu_train_time > cpu_train_time:
            print("⚠️  GPU is slower than CPU - something is wrong!")
        else:
            print("✅ GPU is faster than CPU")

print("\n=== Next Steps ===")
print("1. Run 'nvidia-smi' in another terminal during this script")
print("2. Check if GPU utilization goes above 50% during training")
print("3. If GPU utilization stays low, TensorFlow isn't actually using the GPU")
print("4. Share the output of this diagnostic script")