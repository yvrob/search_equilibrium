# Pebble Bed Reactor Fast Equilibrium Finder

## Using the tool

* Clone repository.
* Make a conda environment with Cerberus linked.
* Locate the HxF tools folder for the Serpent path.
* Copy the Serpent input files into the same folder as the script.
* Modify the input variables in the script.
* Run with `python search_equilibrium.py`

## Script Explanation

(The following description was temporarily made by ChatGPT, using the main script in the prompt)

This repository contains a Python script specifically designed for simulating a Pebble Bed Nuclear Reactor using the Serpent Monte Carlo code, enhanced with the Cerberus framework. The script is structured to comprehensively manage and execute the simulation, encompassing initialization, time-stepping, data monitoring, and solver interaction.

### Import Modules and Constants Definition
The script begins by importing necessary Python libraries such as `pandas`, `numpy`, `matplotlib`, and specific modules from the Cerberus solver framework. It then defines a set of constants that are crucial for the simulation, including parameters like transport, activation, and step sizes.

### Classes Definition
Two main classes, `Simulation` and `Pebble_bed`, form the core of the script:

- **Simulation Class**: Manages the entire simulation process. It initializes the Serpent/Cerberus simulation, handles file parsing, conducts simulation time-stepping, and manages data for solver interaction. This class plays a crucial role in setting up the simulation environment, managing the execution flow, and extracting simulation results.

- **Pebble_bed Class**: Focused on the pebble bed reactor's specific characteristics. It includes functionalities for reading pebble bed data, managing pebble attributes, slicing and clipping pebble bed distributions, and plotting 2D and 3D visualizations of the reactor's state.

### Utility Functions
A series of utility functions are defined for tasks like generating link matrices for material transfers, plotting flux distributions, and handling random assignments within the pebble bed. These functions are essential for managing the complex interactions and transformations of materials within the reactor.

### Initialization
The script initializes the Serpent simulation, setting up the necessary environment and configurations. It involves specifying input files, reactor geometry, material properties, and other essential parameters. The pebble bed object is also created and configured in this section.

### Main Simulation Loop
The core of the script is the main loop where the actual simulation steps are executed. It includes:

- Transport and activation steps to simulate neutron interactions and fuel depletion.
- Adjusting simulation parameters dynamically based on the results, such as neutron count and motion steps.
- Applying convergence criteria to determine the equilibrium state of the reactor.

During each step of the simulation, the script manages the intricate balance of neutron transport, material movement within the reactor, and data extraction for analysis. The loop continues until the predefined convergence criteria are met, ensuring that the reactor's behavior is accurately captured over the simulation period.

### Conclusion and Output
The script concludes by outputting the simulation results, which include detailed data on neutron flux, fuel burnup, and other critical reactor parameters. These outputs are crucial for analyzing the reactor's performance and safety characteristics.

