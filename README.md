# dimersmodel
Dimers model Rnn

This repository is presented as part of the project "Dimer Models Through the Lens of Recurrent Neural Networks" by Jefferson Pule, presented to as part of Physics 437A class. 
Presented on August 23 2021.

In the project we use different approaches to train the RNN. The dimersmodel.py document and the dominstraining.py represent a complete code that trains an RNN by trying to
generate perfect samples of the dimer model. 

In the other hand the dimer_without constraints contains a code to trian the RNN without a few constrains but by implementing an error factor in the Energy. 
With and without the implementation of annealing. 

Similarly we include a only sample code, that only produces ns samples in the different representations (dimers or dominoes).

In all cases the relevant information is given inside of the class Variational Montecarlo, and parameters are to be updated.

In the results folder we have some of the results we obtained. More results will be updated soon. The chapter name represents the type of training and the size. 
Documents named all_nh_{}_size{}x{}x{}_T_{}.txt containg all the information of the trining giving erros, and samples for a training with nh hidden variables size LxxLyx number of samples
and T being the temperature. 
Similarly we have particular data saved, this data includes the loss function (loss), the free energy (free), the energy (ener), the variance of the energy (var), and the binder cumulants for
dimers symmetry breaking (bdsb) and pair symmetry breaking (bpsb). As well as the last set of samples generated as a file samples_nh_{}_size{}x{}x{}_T_{}_epoch_{}.pt.

This folder is to be updated to include all the information necessary for the project. 

