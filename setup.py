from setuptools import setup, find_packages

requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        requirements.append(line.strip())

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
    install_requires = requirements
)
