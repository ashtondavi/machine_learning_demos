import setuptools

setuptools.setup(
    name='machine_learning_demos',
    version='0.1.0',
    author='David Ashton',
    url='https://github.com/ashtondavi/machine_learning_demos',
    description='Python package of several machine learning demos',
    long_description=open('README.txt').read(),
    package_dir = {"": "machine_learning_demos"},
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.23.1"],
    python_requires='>=3.7',
)