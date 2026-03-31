import sys
import tensorflow as tf

print("Python executable:", sys.executable)
print("TensorFlow imported from:", getattr(tf, "__file__", "NO FILE"))
print("TensorFlow Version:", getattr(tf, "__version__", "VERSION NOT FOUND"))
print("Built with CUDA:", tf.test.is_built_with_cuda() if hasattr(tf, "test") else "test module missing")
print("GPU Available:", tf.config.list_physical_devices('GPU') if hasattr(tf, "config") else "config missing")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')) if hasattr(tf, "config") else 0)