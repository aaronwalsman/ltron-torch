import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ltron-torch",
    version="0.0.0",
    install_requires = [
        'ltron', 'tqdm', 'numpy', 'pyquaternion', 'tensorboard'],
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description='LTRON Torch Experiments"',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/ltron-torch",
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts' : [
            'train_break_and_make=ltron_torch.scripts.train_break_and_make:main'
        ]
    }
)
