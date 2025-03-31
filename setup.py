from setuptools import setup

setup(
    name='torchrun_jsc',
    python_requires='>=3.6,<4',
    version='0.0.18',
    install_requires=[
        'packaging',
        # PyTorch 1.9 introduced the `torchrun` API.
        #
        # Since that version is already quite old in the fast-moving
        # field of deep learning and this package is concerned only with
        # fixing `torchrun` issues, we do not support earlier versions
        # with the old `torch.distributed.launch` API.
        #
        # We allow any future PyTorch versions but print a warning if a
        # not officially supported version is used.
        'torch>=1.9',
    ],
    entry_points={
        "console_scripts": [
            "torchrun_jsc = torchrun_jsc:main",
        ],
    },
    author='Jan Ebert',
    author_email='ja.ebert@fz-juelich.de',
    description=(
        'Fixed version of `torchrun` on Jülich Supercomputing Centre. '
        'Requires Slurm usage.'
    ),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    url='https://github.com/HelmholtzAI-FZJ/torchrun_jsc',
)
