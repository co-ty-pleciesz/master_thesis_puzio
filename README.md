# Investigation of Brain Information Transfer in Intention Decoding Using Multivariate Transfer Entropy

Repository purpose is to keep all programs, articles, tools and informations used to write my master thesis.






## Toolkit that I use

 - [The Information Dynamics Toolkit xl (IDTxl)](https://github.com/pwollstadt/IDTxl)
    P. Wollstadt, J. T. Lizier, R. Vicente, C. Finn, M. Martinez-Zarzuela, P. Mediano, L. Novelli, M. Wibral (2018). IDTxl: The Information Dynamics Toolkit xl: a Python package for the efficient  analysis of multivariate information dynamics in networks. Journal of Open Source Software, 4(34), 1081. https://doi.org/10.21105/joss.01081.

## Theoritical background

 - [Theoretical Introduction to IDTxl toolkit ](https://github.com/pwollstadt/IDTxl/wiki/Theoretical-Introduction)
 - [Information theory - Joseph Lazier lecture on youtube](https://youtube.com/playlist?list=PLOfPLLxr5gsVLSlmzcMnsFANb-uWkArby&si=CmDHGDc9H7kVCQrs)

   This playlist presents video lectures from a course on using information theory for analysing complex systems, with particular focuses on:
    1. Measures of information dynamics: how information is processed in complex systems, including measures of information storage and transfer;
    2. Empirical data analysis using the JIDT open-source software - http://github.com/jlizier/jidt (which is java version of what i use)

## Other research that uses this toolkit

 - [Brain Connectivity Analysis for EEG-Based Face Perception Task](https://app.dimensions.ai/details/publication/pub.1169254368)
 - [Revealing non-trivial information structures in aneural biological tissues via functional connectivit](https://app.dimensions.ai/details/publication/pub.1171650046)





## Key Terminology: 
* `Multivariate Transfer Entropy` - Multivariate Transfer Entropy (MTE) is a measure of directional information flow between multiple variables in a dynamic system, extending the classical Transfer Entropy (TE) to account for multivariate interactions. Unlike pairwise TE, MTE quantifies the amount of information transferred from one or more source variables to one or more target variables, while conditioning on the state of other variables in the system, thereby capturing more complex dependencies and interactions.
* 'mTE Value': The estimated amount of information that the past of a given source XiXi​ provides about the future state of the target YnYn​, conditioned on both the past of the target and the other relevant sources in the set ZZ. This value quantifies how much knowledge of the source helps predict the target's next state, beyond the target's own history.
* `Difference between replications in mTe and Multiscale` - 
In mTE multiple replications of a single sample can be collected (a replication is intended as a physical or temporal copy of the process, or a single experimental trial);
Samples can be collected both over time and over replications to form an ensemble of time series, which is treated as a 3D structure
In contrast, multiscale analysis looks at the same data across different scales, such as various time resolutions, to capture patterns or behaviors that emerge at different levels of detail. Multiscale analysis is about zooming in or out on the temporal or spatial aspects of a single dataset, while the ensemble of time series represents data collected from multiple replications of the same process.

 So, while ensemble of time series focuses on variability across different realizations, multiscale focuses on examining patterns within a single realization across multiple scales. Both methods offer insights into different dimensions of the data but from distinct perspectives.
* 
