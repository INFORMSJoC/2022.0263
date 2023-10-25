# 2022.0263

### Cite 

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2022.0263

https://doi.org/10.1287/ijoc.2022.0263.cd

Below is the BibTex for citing this snapshot of the respoitory.

@article{sherzer2023SupervisedGG1,
  author =        {Baron, Opher and Krass, Dmitry and Senderovich, Arik and Sherzer, Eliran},
  publisher =     {INFORMS Journal on Computing},
  title =         {Supervised ML for Solving the GI/GI/1 queue},
  year =          {2023},
  doi =           {10.1287/ijoc.2022.0263.cd},
  url =           {https://github.com/INFORMSJoC/2022.0263},
}  


## The following package includes:

### 1. Sampling code of a G/G/1 queue.
### 2. The Neural network prediciton model of a G/G/1 queue.
### 3. G/G/1 data sets.


### Sampling code of a G/G/1 queue:
This part contains a code of a G/G/1 code, where one can use our samlping algorithm for the inter-arrival and service time distributions and then compute the true steady-state probabilites of the number of customers in the system.

The user may control the following configurations: --num_examples: number of G/G/1 batches to sample, where batch contains multiple G/G/1 queues. --batch_size: number of examples in a G/G/1 queue. At the end, the system saves two files for each batch. One file contains the inter-arrival and service time moments and the other contain the steady-state probabilities. --num_moms: number of inter-arrival and service time moments are saved. --data_path: The directory path where we wish the data will be saved. --max_utilization: the maximum utilization of the G/G/1 queue.

### Neural network prediciton model:
In this part, the user is required to give 5 inter-arrival and service time moments and the program's output is a 500 size vector of the steady-state probability. Whereas, the i^th location represent the number of customers n the system under steady-state. The program also plots a bar chart of the disitrbution (outputs only the first 30 values, but the user can change it).

The user is required to insert the first 5 inter-arrival and service time moments in lines 120 and 121, respectively.

Currently, there is an existing example, which the user is encourages to replace.

### G/G/1 data:
The folder of the data can be downloaded from this link:

https://utoronto-my.sharepoint.com/:f:/g/personal/eliran_sherzer_rotman_utoronto_ca/EqnABrSRlz1MoXW6Duyr6zEBRzFWB6QZbBieLG0X3Dhetg?email=eliran.sherzer%40rotman.utoronto.ca&e=ajaRfL

The folder contains python pickle file. There are 4 files:

train_moms_valid_qa.pkl
train_ys_valid_qa.pkl
train_ph_moms.pkl
train_ph_ys.pkl
Files 1 and 2 represent a small G/G/1 data set.File 1 is the moments and file 2 is the steay-state probabilites. Files 3 and 4 represent a large G/G/1 data set.File 3 is the moments and file 4 is the steay-state probabilites.

