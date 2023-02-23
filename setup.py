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
            'train_break_and_make_bc='
                'ltron_torch.train.break_and_make_bc:train_break_and_make_bc',
            'train_break_and_make_dagger='
                'ltron_torch.train.break_and_make_dagger:'
                'train_break_and_make_dagger',
            'train_break_and_estimate_dagger='
                'ltron_torch.train.break_and_estimate_dagger:'
                'train_break_and_estimate_dagger',
            'train_classify_dagger='
                'ltron_torch.train.classify_dagger:train_classify_dagger',
            'train_edit_bc='
                'ltron_torch.train.edit_bc:train_edit_bc',
            'train_edit_dagger='
                'ltron_torch.train.edit_dagger:train_edit_dagger',
            'plot_break_and_make_bc='
                'ltron_torch.train.break_and_make_bc:plot_break_and_make_bc',
            'eval_break_and_make='
                'ltron_torch.train.break_and_make_eval:break_and_make_eval',
            'train_blocks_bc=ltron_torch.train.blocks_bc:train_blocks_bc',
            'ltron_break_and_make_dataset='
                'ltron_torch.dataset.break_and_make:'
                'ltron_break_and_make_dataset',
            'train_select_connection_point='
                'ltron_torch.train.select_connection_point:'
                'train_select_connection_point',
            'train_break=ltron_torch.train.break:train_break',
            'train_make=ltron_torch.train.make:train_make',
        ]
    }
)
