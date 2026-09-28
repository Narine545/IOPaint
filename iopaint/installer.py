import platform
import subprocess
import sys


def install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def install_plugins_package(directml: bool = False):
    """Install the optional dependencies used by the plugins.

    ``directml=True`` installs the DirectML build of onnxruntime, which lets any
    DirectX 12 GPU (including AMD/Intel integrated graphics) run the background
    removal models on Windows.
    """
    if directml:
        if platform.system() != "Windows":
            raise RuntimeError("onnxruntime-directml is only available on Windows")
        # onnxruntime and onnxruntime-directml cannot coexist
        subprocess.call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"]
        )
        install("onnxruntime-directml")
    else:
        install("onnxruntime")

    # rembg pulls onnxruntime by default, avoid overwriting the directml build
    install("rembg")
    # closed form matting ('--remove-bg-alpha-matting quality')
    install("pymatting")
