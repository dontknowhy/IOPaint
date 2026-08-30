import setuptools
from pathlib import Path

# --- Web-app assets -------------------------------------------------------
# The frontend is built with npm and the dist/ output is copied into
# iopaint/web_app/ by publish.sh (or by the developer manually).
# When that directory doesn't exist (e.g. a raw clone + `pip install ./`),
# we gracefully fall back to an empty list so that the install still works
# (the CLI/API will function, just without the bundled web UI).
_web_app_dir = Path("iopaint/web_app")
if _web_app_dir.is_dir():
    package_files = [str(p.relative_to("iopaint")) for p in _web_app_dir.rglob("*") if p.is_file()]
else:
    package_files = []

# --- Static data files shipped inside the iopaint package -----------------
package_files += [
    "model/anytext/ocr_recog/ppocr_keys_v1.txt",
    "model/anytext/anytext_sd15.yaml",
    "model/original_sd_configs/sd_xl_base.yaml",
    "model/original_sd_configs/sd_xl_refiner.yaml",
    "model/original_sd_configs/v1-inference.yaml",
    "model/original_sd_configs/v2-inference-v.yaml",
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def load_requirements():
    requires = []
    with open("requirements.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requires.append(line)
    return requires


# https://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files
setuptools.setup(
    name="IOPaint",
    version="1.6.0",
    author="PanicByte",
    author_email="cwq1913@gmail.com",
    description="Image inpainting, outpainting tool powered by SOTA AI Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sanster/IOPaint",
    packages=setuptools.find_packages("."),
    package_data={"iopaint": package_files},
    install_requires=load_requirements(),
    python_requires=">=3.8",
    entry_points={"console_scripts": ["iopaint=iopaint:entry_point"]},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
