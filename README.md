# `torchrun_jsc`: `torchrun` on Jülich Supercomputing Centre

Due to [various issues with the `torchrun`/`torch.distributed.run` API
on Jülich Supercomputing Centre (JSC)
systems](https://github.com/pytorch/pytorch/issues/73656), this
package provides a new launcher called `torchrun_jsc` that wraps the
old one as a drop-in replacement. This package really just provides a
fixed `torchrun`, so contrary to the name of this package, it is
portable across all machines. In other words, `torchrun_jsc` supports
a superset of machines that `torchrun` supports, thus there is no need
to special-case `torchrun_jsc` in your scripts.

The only requirement for its usage is Slurm and a PyTorch version ≥1.9
(because earlier versions do not have the `torchrun` API implemented).
The package hopes that the current solution is forward-compatible, but
will emit warnings if a PyTorch version ≥3 is used.

## Installation

### PyPI

```shell
python -m pip install torchrun_jsc
```

### Source

```shell
python -m pip install git+https://github.com/HelmholtzAI-FZJ/torchrun_jsc.git
```

## Usage

Modify your execution like the following:

Old
```shell
torchrun [...]
# or
python -m torch.distributed.run [...]
```

New
```shell
torchrun_jsc [...]
# or
python -m torchrun_jsc [...]
```

Please remember to use `srun` to start your Python processes,
otherwise necessary Slurm variables will not be set.

## How does it work?

First, the `torchrun_jsc` launcher will always patch `torchrun`'s
`--rdzv_conf` argument's `is_host` configuration so that the correct
process is recognized as the host process for setting up the
communication server.

After that, depending on your PyTorch version, there are various modes
of functionality to achieve that the correct address is used for
rendezvousing:
1. PyTorch ≥3: Monkey-patch the function used to obtain the rendezvous
   hostname and the function setting up rendezvous metadata and emit a
   warning.
2. PyTorch ≥2.4 <3: Monkey-patch the function used to obtain the
   rendezvous hostname and the function setting up rendezvous
   metadata.
3. PyTorch ≥1.9 <2.4: Monkey-patch the function used to obtain the
   rendezvous hostname.
4. PyTorch <1.9: If this package is somehow installed for a
   non-matching PyTorch version, it will error out because the
   `torchrun` API does not exist in these versions.
