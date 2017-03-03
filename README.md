# An autonomic computing approach for selecting dynamic loop scheduling and improving performance of scientific applications

---

## Abstract
Scientific applications are often time-consuming, complex and irregular. They are frequently deployed on shared high performance, parallel and distributed environments. Two main reasons can contribute to the poor performance of scientific applications during their execution: the irregularity of the algorithms associated to these applications and the unpredictable variations of the targeted high performance computing (HPC) infrastructures.

This master project is interested in time-stepping scientific applications (TSSA). 
It studies the Dynamic Loops Scheduling (DLS) methods that target parallel loops in TSSA and the reinforcement learning (RL) techniques that could be applied in order to adapt DLS to the irregular behavior of TSSA. It also introduces a new criterion to evaluate the performance of TSSA: the robustness.
A simulation environment is developed in order to evaluate the different RL techniques with the robustness criterion on different ranges of TSSA. 
The simulation results underscore the effectiveness of using an autonomic computing approach in the selection of dynamic loop scheduling algorithms for performance optimization of time stepping scientific applications. 

## What is this ?
This is the code repository of this master project. It contains installation instructions, tools and source codes to run massively parallel applications (MPI) on a generic simulation framework (SimGRID v3.13) using SMPI. Besides, the parallel application is improved using DLS and RL. This project is on version 1.0, but it is still under development. 

## Content
This project is developed in C++11, this project thus contains the source code (in C++11), some tools to ease the execution of test batches (in Python2.7) and a Makefile.

### Source code
- Reinforcement Learning library (QLearning, QVLearning, Sarsa, ExpectedSarsa, DoubleQLearning)
- Dynamic Loop Scheduling library **DLS**
- Library for computing robustness of DLS strategy
- Source code to simulate MPI application

### Binaries
- **mono_dls**: This binary simulates the execution of a parallel loop using a particular DLS strategy
- **all_dls**: The same behavior as *mono_dls* but with all the DLS strategies. 
- **rl_only**: It is a test application for the RL library.
- **smpi_robustness_with_rl**: It simulates the execution of a parallel loop on SMPI and optimizes the selection of DLS strategy using a particular RL algorithm.

## Installation guide
This project requires several libraries/tools, the list is as follows:

##### Compiler & Interpreter
- g++ (with C++11) 
- Python2.7 (with numpy, matplotlib, futures)

##### SimGRID dependencies
- ccmake / cmake / make
- libboost-all
- doxygen
- openmpi (for debugging only)

##### External libraries
- jsoncpp

In Debian/Linux install you can install the dependencies as follows:
```bash
sudo apt-get update
sudo apt-get install gcc g++ cmake-curses-gui cmake make python2.7 libboost-all-dev mpich python-pip git python-numpy doxygen
sudo pip install pip matplotlib numpy --upgrade
sudo pip install futures
sudo git clone https://gitlab.com/anthony.boulmier.cfpt/master-thesis.git ~/ScientificApplicationOptimization
sudo git clone https://github.com/open-source-parsers/jsoncpp.git ~/jsoncpp 
cd ~/jsoncpp/
sudo python amalgamate.py
cd ~
```

### SimGRID
1. Download SimGRID v3.13 via this [link](https://gforge.inria.fr/frs/download.php/file/35817/SimGrid-3.13.tar.gz)
2. Extract SimGRID in the folder of your choice (you may use `sudo tar -xzf`) 
3. Compile SimGRID (I suggest you to use `sudo ccmake .` at the root of the SimGRID folder). If you choose to use ccmake, you can change the installation folder to `/opt/simgrid` as it you won't need to modify the env. variables of the Makefile.
4. Install SimGRID using `sudo make install`
5. _Optional_: test your installation using `sudo make test`.

### Compile the project
1. Go at the root of the project (it is `cd ~/ScientificApplicationOptimization` if you did not touch anything).
2. `make all` (if you changed the installation folder of SimGRID or JSONCPP, you can compile using it like that: `make all SIMGRID=root_of_the_simgrid_installation JSONCPP_HOME=root_of_jsoncpp` 
3. All the binaries are available in the bin/ folder.

## Get started
SimGRID requires information about the "simulated" infrastructure:

1. A platform on which the code will run. You can generate a platform using the script `tools/generate_platform.py`.
2. A hostfile (just a list of the simulated hosts). You can generate the hostfile along the creation of the platform. 
3. Number of processors

There are two ways to go:
1. Directly use the binaries via SMPI and `/opt/simgrid/bin/smpirun`. All the binaries have a helper, you can access it via the `--help` option.  
2. Call the binaries through the tools available in the `tools` folder. All the tools have a helper, you can access it via the `--help` option.

If you want to use the binaries directly, I suggest you to read the [SimGRID's documentation](http://simgrid.gforge.inria.fr/simgrid/3.13/doc/index.html).

### Tools
- To generate platforms and hostfiles: `tools/generate_platform.py`
- To generate the effect of the infrastructure on a simulated application: `tools/generate_perturbations.py`
- To generate the ideal parallel time for a given application on a particular infrastructure: `tools/generate_ideal_tpar.py`
- To execute the complete time-stepping application improvement using RL and Robustness: `tools/launcher.py`
- To plot the results obtained via the time-stepping application improved: `tools/plot.experiment.py`

If you have any questions, please ask: [anthony.boulmier@hesge.ch](mailto:anthony.boulmier@hesge.ch?subject="Questions about your master thesis")


