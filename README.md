# RGP-GAN
Exploration of GAN for synthetic motor imagery data.

## Dataset description

A motor imagery movement experiment was conducted for left and right hands with 52 participants (19 females, mean age = 24.8 ± 3.86 years). Participants s20 and s33 were both-handed, and the other 50 participants were right-handed. The data was collected using Biosemi ActiveTwo system (64 Ag/AgCl active electrodes), with f_s=512 Hz and 10-10 montage.

Before the actual MI experiment, each participant was asked to move their fingers, starting from the index finger and proceeding to the little finger and touching each to their thumb within 3 seconds after onset.

Then, during the experiment, participants were asked to imagine the hand movement depending on the instruction given. 5/6 runs were performed during the experiment. After each run, the classification accuracy for this respective run was calculated and outputed to the participant to increase motivation. Between runs, a maximum 4-minute break was given depending on the participant's demands.

More: https://academic.oup.com/gigascience/article/6/7/gix034/3796323

## Preprocessing

Analogous to the manuscript entitled **Riemannian Geometry-Preserving Variational Autoencoder for MI-BCI Data Augmentation**. Files like `data_utils.py`, `riemannian_parallel_transport.py` and `spd_utils.py` were adopted from the corresponding repository.

## Vizualisaiton

`vizualizer.py` was adopted from the entitled manuscript.