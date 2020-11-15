from setuptools import setup, find_packages

setup(
    name = 'naszilla',
    version = '1.0',
    author = 'Colin White, Willie Neiswanger',
    author_email = 'crwhite@cs.cmu.edu',
    description = 'python framework for NAS algorithms on benchmark search spaces',
    license = 'Apache 2.0',
    keywords = ['AutoML', 'NAS', 'Deep Learning'],
    url = "https://github.com/naszilla/naszilla",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires = [
        'tensorflow-gpu==1.14.0',
        'tensorflow==1.14.0',
        'torch==1.5.0',
        'torchvision==0.6.0',
        'nas-bench-201==1.3',
        'pybnn'
        'autograd>=1.3'
        'click'
        'Cython'
        'ConfigSpace==0.4.12'
        'ipython'
        'lightgbm>=2.3.1'
        'matplotlib'
        'numpy'
        'pandas'
        'pathvalidate'
        'Pillow>=7.1.2'
        'psutil'
        'scikit-image'
        'scikit-learn>=0.23.1'
        'scipy'
        'seaborn'
        'statsmodels'
        'tensorboard==1.14.0'
        'tensorflow-estimator'
        'tqdm'
        'xgboost'
    ]
)
