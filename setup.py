from setuptools import find_packages
from setuptools import setup

setup(
    name='gan-dogs',
    version='1.0.0',
    description='Generative Dog Images with BigGan.',
    author='caciolai, bondanza',
    license='MIT License',
    url='https://github.com/caciolai/Generative-Dog-Images-with-BigGan',
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},   # tell distutils packages are under src
    python_requires='>=3.7.9',
    install_requires=[
        'albumentations',
        'keras',
        'matplotlib',
        'numpy',
        'opencv-python',
        'pandas',
        'scikit-learn',
        'seaborn',
        'tensorflow>=2.3.0',
        'tqdm',
        'typeguard'
    ]
)