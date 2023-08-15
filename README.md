DACBench is a benchmark library for Dynamic Algorithm Configuration.
Its focus is on reproducibility and comparability of different DAC methods as well as easy analysis of the optimization process.

You can try out the basics of DACBench in Colab [here](https://colab.research.google.com/drive/1XQhJM2ErvDMu7eSSWFFEedGdw8GRChIE?usp=sharing) without any installation.
Our [examples](https://github.com/automl/DACBench/tree/main/examples) in the repository should give you an impression of what you can do with DACBench and 
our [documentation](https://dacbench.readthedocs.io/) should answer any questions you might have. 

You can find baseline data of static and random policies for a given version of DACBench on our [project site](https://www.tnt.uni-hannover.de/en/datasets/dacbench/).


## Installation
We recommend installing DACBench in a virtual environment:

```
conda create -n dacbench python=3.10
conda activate dacbench
pip install dacbench
```

Instead of using pip, you can also use the GitHub repo directly:
```
git clone https://github.com/automl/DACBench.git
cd DACBench
git submodule update --init --recursive
pip install .
```
This command installs the base version of DACBench including the three small surrogate benchmarks and the option to install the FastDownward benchmark.
For any other benchmark, you may use a singularity container as provided by us (see next section) or install it as an additional dependency. As an example, 
to install the SGDBenchmark, run:

```
pip install dacbench[sgd]
```

To use FastDownward, you first need to build the solver itself. We recommend using
cmake version 3.10.2. The command is:
```
./dacbench/envs/rl-plan/fast-downward/build.py
```

You can also install all dependencies like so:
```
pip install dacbench[all,dev,example,docs]
```

## Containerized Benchmarks

DACBench can run containerized versions of Benchmarks using Singularity containers to isolate their dependencies and make reproducible Singularity images. 


### Building a Container

For writing your own recipe to build a Container, you can refer to `dacbench/container/singularity_recipes/recipe_template`  

Install [Singularity](https://docs.sylabs.io/guides/3.6/user-guide/) and run the following to build the (in this case) cma container

```bash
cd dacbench/container/singularity_recipes
sudo singularity build cma cma.def
```

## Additions for Master's project
Population size adaptation environment and benchmark
`modcma_pop_size.py`
`modcma_popsize_benchmark.py`

Dummy environment and benchmark to control population size with PSA-CMA-ES (Kouhei Nishida and Youhei Akimoto. 2018. PSA-CMA-ES: CMA-ES with population size adaptation.)
`artificial_modcma.py`
`artificial_modcma_benchmark.py`

Script to train RL agent
`main.py`

Script to evaluate RL policy against PSA-CMA-ES
`test.py`

Plotting functionality
`plot.py`


## Citing DACBench
If you use DACBench in your research or application, please cite us:

```bibtex
@inproceedings{eimer-ijcai21,
  author    = {T. Eimer and A. Biedenkapp and M. Reimer and S. Adriaensen and F. Hutter and M. Lindauer},
  title     = {DACBench: A Benchmark Library for Dynamic Algorithm Configuration},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence ({IJCAI}'21)},
  year      = {2021},
  month     = aug,
  publisher = {ijcai.org},
```
