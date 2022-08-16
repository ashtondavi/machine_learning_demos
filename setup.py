import setuptools

setuptools.setup(
    name='demos',
    version='0.1.0',
    author='David Ashton',
    url='https://github.com/ashtondavi/machine_learning_demos',
    description='Python package of several machine learning demos',
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=[],
    python_requires='>=3.7',
)