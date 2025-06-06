The objective of this code is to develop an Artificial Neural Network for analysis of PIXE data of historical silver coins. The repository is organized in:

•	Information on the coins is given in the <publications> folder and in https://doi.org/10.1016/j.microc.2016.12.002.

•	The experimental data is in folder <experimental_data>. The folder also has the file <cal1.txt> with the energy calibration and an excel file with a list of the data files and information on line yields and derived concentrations.

•	A set of fortran routines for data augmentation in folder <data_augmentation_code>. This is in file <silver data augmentation source project 2025-04-18.zip>, which also includes the project files for the Intel Fortran Visual Studio Project.

•	The data augmentation routine works on the data provided in the <experimental_data> folder.

•	The output of the data augmentation routine is in files <silver data augmentation output X 00.zip> to <silver data augmentation output X 10.zip>, in folder <data_augmentation_code> (X is the version, currently j and m). The options with which the routine was ran are given in the file <j_options.txt> contained in that zip.

•	The ANN was developed and tested in Visual Studio Code. The code is in folder <ANN_code>. It uses Pandas, Numpy and Tensorflow 2.2 libraries, and it is given in file <silver 2025-04-19 ANN.zip>. It uses as input some of the files contained in <silver_data_augmentation_output_X.zip>, namely <X_data_iiiii.dat> and <X_synth_iiiii.dat>, where iiiii is a five digit integer, <X_in_data.csv> and <X_out_data.csv>.

•	The output of the ANN, for the training set and the test set, are given in the files <train_X.xslx> and <test_X.xslx>, contained in file <silver 2025-04-17 X results.xslx>. Some data were not used in the training or testing at all, nevertheless the outputs of the ANN are calculated and are given in file <error_j.xslx> (not for m data). Note that the training process uses random numbers, and each run of the code will produce slightly different results.

•	Different code versions and their output are organized by date. If a code version and a data file version have the same date, they correspond to the same version.
