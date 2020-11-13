from setuptools import setup, find_packages

setup(
    name = 'naszilla',
    version = '1.0.1',
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
    ]
)
