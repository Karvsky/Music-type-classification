import os
import pathlib

# 1. Konfiguracja ścieżek dla NVIDIA wewnątrz venv
# Szukamy folderów bin wewnątrz venv/Lib/site-packages/nvidia
nvidia_base = pathlib.Path(os.getcwd()) / 'venv' / 'Lib' / 'site-packages' / 'nvidia'

if nvidia_base.exists():
    # Przeszukujemy podfoldery takie jak 'cudnn', 'cublas', 'cuda_runtime'
    for subfolder in nvidia_base.iterdir():
        bin_dir = subfolder / 'bin'
        if bin_dir.exists():
            os.add_dll_directory(str(bin_dir))
            print(f"✅ Aktywowano bibliotekę: {subfolder.name}")

# 2. Dopiero teraz importujemy TensorFlow
import tensorflow as tf

print("\n--- STATUS GPU ---")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"🚀 SUKCES: TensorFlow widzi GPU: {gpu_devices}")
else:
    print("⚠️ Nadal brak GPU. Spróbuj zrestartować VS Code.")