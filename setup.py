import setuptools
from pathlib import Path

package_files = Path("iopaint/web_app").glob("**/*")
package_files = [str(it).replace("iopaint/", "") for it in package_files]
package_files += ["model/anytext/ocr_recog/ppocr_keys_v1.txt"]
package_files += ["model/anytext/anytext_sd15.yaml"]
package_files += ["model/original_sd_configs/sd_xl_base.yaml"]
package_files += ["model/original_sd_configs/sd_xl_refiner.yaml"]
package_files += ["model/original_sd_configs/v1-inference.yaml"]
package_files += ["model/original_sd_configs/v2-inference-v.yaml"]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def load_requirements(requirements_file_name="requirements.txt"):
    requires = []
    with open(requirements_file_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requires.append(line)
    return requires


EXTRAS = {
    "diffusion": load_requirements("requirements-diffusion.txt"),
    "plugins": load_requirements("requirements-plugins.txt"),
    "webconfig": load_requirements("requirements-webconfig.txt"),
}
EXTRAS["all"] = sorted({dep for deps in EXTRAS.values() for dep in deps})


# https://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files
setuptools.setup(
    name="IOPaint",
    version="1.7.0",
    author="PanicByte",
    author_email="cwq1913@gmail.com",
    description="Image inpainting, outpainting tool powered by SOTA AI Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sanster/IOPaint",
    packages=setuptools.find_packages("."),
    package_data={"iopaint": package_files},
    install_requires=load_requirements(),
    extras_require=EXTRAS,
    python_requires=">=3.9",
    entry_points={"console_scripts": ["iopaint=iopaint:entry_point"]},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
