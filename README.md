# GUNTAM

[![codecov](https://codecov.io/gh/Corentin-Allaire/GUNTAM/branch/main/graph/badge.svg)](https://codecov.io/gh/Corentin-Allaire/GUNTAM)

This generic library is intended to explore the use of Transformer networks for trajectory reconstruction. 
In particular, it focuses on charged particle tracking, integrating with the data from the ACTS project.

For more information, you can refer to the presentation in the Connecting the Dots 2025 conference and the associated proceedings: https://indico.cern.ch/event/1499357/contributions/6628608/ 

<img width="1086" height="622" alt="GUNTAM" src="https://github.com/user-attachments/assets/d01d227a-45bf-4b83-b288-f31ea4ad7eac" />

## Installation 

This library use PDM for easy setup. To install the package, simply run:
```
pdm install
eval "$(pdm venv activate)"
```

## Datasets

This library currently relies on using data obtained with the ACTS library (with the Open Data Detector). The easiest way to get the input data is to run the full_chain_odd.py script and collect the different output CSV files. In total, for the training, four different files will be needed :
- The list of hits in the detector (eventXXXXXXXXX-hits.csv)
- The list of particles in the detector (eventXXXXXXXXX-particles.csv could be used, but for simplicity, we added a ParticleWriter to the DigiParticleSelection to limit the writing to particles of interest).
- A list of space points (eventXXXXXXXXX-spacepoint.csv) is not strictly needed as hits can be used instead, but for more realistic performances, SP are recommended. They can be extracted by adding a SpecePointsWriter to the addSeeding function.
- A map between measurement ID and hit ID is needed to match space points to particle (eventXXXXXXXXX-measurement-simhit-map.csv, only needed if space points are used).
  
To ease the simulation, we recommend running multiple instances of full_chain in parallel (we launch 500 instances with 100 ttbat events each). At the time of writing, the Calorimeter is added by default to the ODD; removing it from the XML file greatly speeds up the simulation.

The data is expected to be different, in a directory called "odd_full_chain_X", in our case, X goes up to 500, and each directory contains $100 \times 4$ CSV files.    


## Data Preprocessing

The first preprocessing pass can be performed using Read_ACTS_Csv.py. This file will read the content of the ACTS CSV file. It can be run using the following command:
```
python Read_ACTS_Csv.py --use-space-point --output-format h5 --file-number "$i" --dir-start "$start" --dir-end "$end" --input-path"$path"
```

Among the options, "dir-start" and "dir-end" represent the id of the "odd_full_chain_X" directory you will be processing (and which are expected to be located at "path"), and the resulting file will have the id "i". Creating multiple files is useful, as the Dataloader used in the Transformer training is designed to load input files sequentially as needed, reducing the memory footprint.

I this step, some initial pre-processing will be performed sequentially:
- Creating of matching particle ID for particles and hits
- Particle filtering (removing the one with fewer than a certain amount of hits)
- Hits processing (removal of the one outside the seeding region, duplicate removal, variable computation and renaming)
- Particle processing (variable computation and renaming)
- Space point processing (association with particles, variable computation and renaming)

The result can be written as either a CSV or hdf5 file (we recommend the latter as the can be up to 10 times smaller)

## Training the Transformer

Once the data has been preprocessed, the training can be easily started by calling: 
```
python Train.py --input_path "$input1" --input_tensor_path "input2"  --input_format "h5" --events_per_file 1 --test_fraction 0.2
```
For more information on the different options, use the "--help" option. If this is the first time running the training, the "prepare_tensor" function will be called to perform the second preprocessing pass on the data (in particular, implementing the selected binning strategy). The fully preprocessed data will then be written to file as a PyTorch tensor at "input2". On subsequent runs, if files in "input2" have metadata matching the current configuration, they will be used directly (preprocessing can be forced with the "--recompute_tensor" option). 

Once the training has been concluded, the weight will be saved as an onnx file, the evolution of the model through the epoch can be obtained with the TensorBoard files stored in training_seeding (by calling ```tensorboard --logdir training_seeding/```). Finally, some monitoring plots will be written to PNG showing the seeding efficiency.


