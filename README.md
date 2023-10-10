# SARMA

SARMA (SpatiAl Rectilinear Matrix pArtitioning) is a template-based, header only,
library for spatial rectilinear partitioning.
The main goal of this library to introduce novel **symmetric** rectilinear partitioning
algorithms. 

💻 **Source Code:** [http://github.com/GT-TDAlab/SARMA]  
📘 **Documentation:** [http://gt-tdalab.github.io/SARMA/]  

SARMA is developed by the members of [GT-TDAlab](http://tda.gatech.edu). 

## License

SARMA is distributed under BSD License. For details, see [`LICENSE`](LICENSE.md).

The MMIO library is included as an external program. See [`MMIO LICENSE`](./external/mmio/LICENSE.md) for details.

## Contributors

- [Abdurrahman Yasar](http://cc.gatech.edu/~ayasar3)
- [M. Fatih Balin](https://www.cc.gatech.edu/~mbalin3/)
- [Kaan Sancak](http://www.kaansancak.com/)
- [Xiaojing An](https://xiaojingan.com/)
- [M. Mucahid Benlioglu](http://mucahidbenlioglu.com)
- [Umit V. Catalyurek](http://cc.gatech.edu/~umit)

## Contact

For questions or support [open an issue](../../issues) or contact contributors via <tdalab@cc.gatech.edu>.

## Citation
Citation for the rectilinear partitioners (BibTeX):

```bibtex
    @techreport{Yasar20-ARXIV,
        author =  {Abdurrahman Ya\c{s}ar and M. Fatih Bal{\i}n and Xiaojing An and Kaan Sancak and {\"{U}}mit V. {\c{C}}ataly{\"{u}}rek},
        title = {On Symmetric Rextilinear Matrix Partitioning},
        institution = {ArXiv},
        number = {arXiv:2009.07735},
        url    = {http://arxiv.org/abs/2009.07735},
        month  = {Sep},
        year   = {2020},
        KEYWORDS = {Spatial partitioning, rectilinear partitioning, symmetric partitioning},
    }
```

Citation for the subgradient method rectilinear partitioners (BibTeX):

```bibtex
    @techreport{Balin23-ARXIV,
        author =  {M. Fatih Bal{\i}n and Xiaojing An and Abdurrahman Ya\c{s}ar and {\"{U}}mit V. {\c{C}}ataly{\"{u}}rek},
        title = {SGORP: A Subgradient-based Method for d-Dimensional Rectilinear Partitioning},
        institution = {ArXiv},
        number = {arXiv:2310.02470},
        url    = {http://arxiv.org/abs/2310.02470},
        month  = {Oct},
        year   = {2023},
        KEYWORDS = {Spatial partitioning, rectilinear partitioning, symmetric partitioning},
    }
```

## How to build

Create a `build` directory and run `cmake` and `make` from there like

    mkdir build
    cd build
    cmake ..
    make -j


### Mac OS X

In order to properly set your compiler, especially if you are using
external package managers like `brew`, we recommend running `cmake` 
with the following options:

    cmake \
        -D CMAKE_OSX_SYSROOT="/" \
        -D CMAKE_OSX_DEPLOYMENT_TARGET="" \
        -D CMAKE_C_COMPILER=gcc-9 \
        -D CMAKE_CXX_COMPILER=g++-9 \
        ..


### Dependencies

`cmake (version >= 3.13)` and `gcc (version >= 7.0.0)` are used to build sarma.

For gcc version 9.0.0 and greater this project depends on Intel's TBB library
because gcc depends on it for its parallel stl implementation. For other gcc versions
parallel execution is disabled.

You can install the latest version with your favorite package manager or 
you could go to [tbb releases](https://github.com/oneapi-src/oneTBB/releases) 
and download the latest release.
[This one](https://github.com/oneapi-src/oneTBB/archive/v2020.2.zip) has been confirmed to work.

To ensure correct installation, we recommend using [build.py](https://github.com/oneapi-src/oneTBB/blob/tbb_2020/build/build.py)
provided by TBB for the installation. It can be simply used as follows from TBB root.

```shell
python3 build/build.py --prefix="<PATH-TO-INSTALL>" --install-libs --install-devel
```
If `tbb` is not found, add environment variable: `export TBB_ROOT=<PATH-TO-INSTALL>`.

The mixed integer program to optimally solve partitioning problems, i.e. the `mip` algorithm, requires
[gurobi 9](https://www.gurobi.com/products/gurobi-optimizer/) to be installed. After installation,
set either of `GUROBI_HOME` or `GUROBI_DIR` environment variables to the root of your installation.

The python bindings depend on pybind11. After installing pybind11, please set `pybind11_DIR` to where
your installation's `pybind11Config.cmake` file is. To install the bindings, use `pip install .` at the
root of the sarma project repository which is where `setup.py` resides. An example
usage can be found below and also under `utils/python_example.py`.

```python
>>> from sarma import *
>>> from scipy.io import mmread
>>> A = mmread('../tests/system/matrices/email-Eu-core.mtx')
>>> print('Shape of the matrix: ', A.get_shape())
Shape of the matrix:  (1005, 1005)
>>> Q = sps(A)
>>> p = nic(A,8)
>>> L = Q.max_load(*p)
>>> print('Max load : ', L)
Max load :  543
>>> print('Cut vectors: ', p)
Cut vectors:  ([0, 61, 113, 168, 249, 339, 434, 551, 1005], [0, 51, 125, 206, 283, 380, 495, 710, 1005])
```

**For tests:**
* [Python 3](https://www.python.org/downloads/) is used for output processing.
* `timeout` is used in tests to ensure timeouts can be cleaned up.
On Mac OS, `timeout` can be installed via `brew install coreutils`. `python3` is used in tests
for outputs processing.

**For Documentation:**

Documentation and API Reference is available [here](https://sarma.github.io) for local generation the following are needed:

* [Python 3](https://www.python.org/downloads/)
* [Pip](https://pip.pypa.io/en/stable/installing/)
* [Doxygen](https://www.doxygen.nl/download.html)

Additionally run `pip install -r docs/requirements.txt` for python dependencies in the project root.

Now running `make docs` in build directory will create documentations under `build/docs/sphinx`

## How to run

You can use `sarma` executable to run partitioning algorithms from command line. 
Use `sarma --help` to see all options. For example, to run Nicol's rectilinear
algorithm from build directory

```shell
sarma --graph ../tests/system/matrices/email-Eu-core.mtx --alg nic -p 4
=========================
Graph     : ../tests/system/matrices/email-Eu-core.mtx
Algorithm : nic
Order     : nat
P         : 4
Q         : 0
Z         : 0
=========================
Cuts:
0	115	254	438	1005	
0	132	283	481	1005	
<<<
1914	1502	1562	1379	
1655	1914	1572	1262	
1626	1621	1923	1248	
1584	1408	1484	1917	
<<<
Max load: 1923
Sparsification time (s): 0
Sparse data structure construction time (s): 0
Partitioning time (s): 0.002112
Load imbalance: 1.20324
Column and row max loads:
1914	1914	1923	1917	
1914	1914	1923	1917
```

If we run SARMA's PAL algorithm for symmetric rectilinear partitioning, the output should
look like this:

```shell
sarma --graph ../tests/system/matrices/email-Eu-core.mtx --alg pal -p 4
=========================
Graph     : ../tests/system/matrices/email-Eu-core.mtx
Algorithm : pal
Order     : nat
P         : 4
Q         : 0
Z         : 0
=========================
Cuts:
0	124	271	458	1005	
<<<
1937	1635	1662	1634	
1545	1929	1607	1421	
1513	1548	1934	1396	
1334	1283	1240	1953	
<<<
Max load: 1953
Sparsification time (s): 0
Sparse data structure construction time (s): 0.008359
Partitioning time (s): 0.007461
Load imbalance: 1.22201
Column and row max loads:
1937	1929	1934	1953
```

### How to run tests
Using ctest for testing.

```shell
cd build
cmake ..
make -j 32
ctest -j 32
```

Some common ctest options:

* To run tests in parallel, add the `-j N` flag, where `N` is the number of tests to run in parallel.
* For selecting tests: you can select tests to run through '-R' option, eg. you can select to run unit tests only by:
    ```shell
    ctest -R 'unit_*'
    ```
  and select all system tests by:
    ```shell
    ctest -R 'sys_*'
    ```
* For more detailed testing output:
    ```shell
    ctest -R 'unit_*' -V
    ```

More and updated option list can be found at [cmake.org on ctest](https://cmake.org/cmake/help/v3.4/manual/ctest.1.html).

#### Unit Tests:

Unit tests are designed for testing correctness of each partitioning function.
Tests uses small examples and are designed to run very fast. 

#### System Tests:

System tests are designed to validate the results of partitioning algorithms. By default,
tests are run on all the matrices under `./tests/system/matrices`
using `./build/sarma` executable with some predertmined arguments.

SHA of results [Cuts, loads, max loads] for each combination of matrices and 
argument is compared against ones from `./tests/system/outputs/valid`

* To reset valid results: Run `ctest` with the environment variable `SARMA_BUILD_VALID` set. For example,
    ```shell
    SARMA_BUILD_VALID=1 ctest
    ```
* To reset testing matrices directory: updating line 10 in:
    ```
    ./tests/system/CMakeLists.txt
    ```
    `cmake` command needs to be rerun after this change.

### Benchmark:

The source code of the benchmark is available under `src/sarma-benchmark.cpp`.
General Usage:
```
sarma-bench [OPTION].. [--help | -h]
Options:                                                        (Defaults)
  -d, --dir    <path>            Directory of the matrices.     ()
  -i, --input  <path>            Input config file.             (stdin)
  -o, --output <path>            Output directory               (stdout)
```
Config format is an input file where each line is one configuration to run. 
Lines starting with hashtag (**#**) are considered as comments and ignored.
Line format:

`graph order algorithm p prob seed load_imbalance time norm_imbalance norm_time`

where each column is deliminated by tab(`\t`);

A sample configuration file for performance tests is available under 
`tests/system/matrices/sarma-bench_configfile.txt`

#### Running Performance Tests:

The systems tests are available under `tests/system/matrices/sarma-bench_configfile.txt`. 
These results are considered as baselines for future changes. 
To run the performance tests `sarma-bench` can be used by running:

```
sarma-bench -d tests/system/matrices/ -i tests/system/matrices/sarma-bench_configfile.txt -o tests/system/matrices/
```
You can monitor the progress via progress bar. Once benchmark is completed, 
the results will be available under `tests/system/matrices`, with proper time stamp.

#### Comparing Performance Between Commits

A fast and easy way to compare performances of two commits is available via 
a `make` command. General usage can be described as:

```
make compare HASH=<commit-hash> CONFIG=<config_file>
```

This command will compare the head commit with the given hash, and configuration file.

## Using SARMA as a library

There are two ways to use Sarma as a library. First, you can copy SARMA repository
into your own project tree. This can be done using a git submodule.
If it is a cmake project, then you can update  your CMakeLists.txt:

```
add_subdirectory(sarma EXCLUDE_FROM_ALL)
add_executable(foo ...)
target_link_libraries(foo SARMA::libsarma)
```

The other method that we encourage is to install SARMA outside your project and
import SARMA as a package. To install SARMA:

```
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
make -j
make install
```

Then an example CMakeLists.txt can be as follows:

```
cmake_minimum_required (VERSION 3.13)
project (foo)
set(CMAKE_CXX_STANDARD 17)
find_package(SARMA REQUIRED)
add_executable(foo foo.cpp)
target_link_libraries(foo SARMA::libsarma)
```

And an example c++ code that uses SARMA as a library that loads and sparsifies
a matrix and partitions it, is provided below. For detailed usage please
see our documentation.

```c++
#include <iostream>
#include "sarma.hpp"

int main(int argc, char** argv){
    std::string file_name = argv[1];
    unsigned int number_of_cuts = atoi(argv[2]);

    auto A_org = std::make_shared<sarma::Matrix<unsigned int, unsigned int>>(file_name, false);

    auto A_sp = std::make_shared<sarma::Matrix<unsigned int, unsigned int>>(
        A_org->sparsify(sarma::utils::get_prob(A_org->NNZ(), number_of_cuts, number_of_cuts, 100.), 4254)
    );

    A_sp->get_sps();

    auto cuts = sarma::probe_a_load::partition<unsigned int, unsigned int>(*A_sp, number_of_cuts);

    for (size_t i = 0; i <= number_of_cuts; i++){
        std::cout << cuts[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

## Primary tested compilers and architectures

Primary tested compilers and systems are:

* GCC 7.4.0, 8.3.0, 9.2.0, 10.1.0, x86, Red Hat 7.6
* GCC 7.4.0, 8.4.0, 9.2.0, 10.1.0, x86, Ubuntu 16.03
* GCC 7.4.0, powerpc64, Red Hat 7.6
* GCC 7.5.0, x86, Ubuntu 18.04
* GCC 7.5.0, x86, CentOS 7.7.1908
* GCC 8.3.1, 10.2.0, x86, CentOS 8.2.2004
* GCC 9.3.0, x86, macOS Mojave 10.14.6
* GCC 9.3.0, x86, macOS Catalina 10.15.5
