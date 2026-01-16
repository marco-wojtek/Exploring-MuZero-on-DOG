import jax

# Print JAX version
print("JAX version:", jax.__version__)

# Print available devices
devices = jax.devices()
print("Available JAX devices:")
for device in devices:
    print(f"  - {device}")

# Check if any GPU is available
if any(device.platform == 'gpu' for device in devices):
    print("JAX is using GPU.")
else:
    print("JAX is using CPU.")