# Wishart Processes and MultidimensionalStochastic Volatility Model

This repository provides with the simulation of CIR process and Wishart process, who are frequently used to describe a stochasticvolatility. 
It provides also two derived pricing model: Sufana-Gourieroux modelm, the extension on multi-dimension of the Heston model) and Fonseca model, who cnsiders assets return dynamics linearlycorrelated with the covariance matrix and performs better by conserving the smile and skew effects of implied correlation.


### Citings
The simulation of CIR process is based on [1] [CIR](https://hal.archives-ouvertes.fr/hal-00143723v5/document), and the simulation of Wishart process if based on [2] [Wihsart](https://projecteuclid.org/journals/annals-of-applied-probability/volume-23/issue-3/Exact-and-high-order-discretization-schemes-for-Wishart-processes-and/10.1214/12-AAP863.full).

The Gourieroux-Sufana model is proposed by [3] [GS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=757312) on 2005.
The Fonseca model is proposed by [4] [Fonseca](https://link.springer.com/content/pdf/10.1007/s11147-008-9018-x.pdf), which is a more complicated model than GS model. 

The tutor of this work is Aurélien Alfonsi.

### Orginasition of codes

* In the file **test.py**, you can find all our test codes and principal API. If you want to use this simulation, this could help a lot.
* In the part **CIR** and **Wishart**, you can find codes to simulation CIR and wishart process.
* In the part **application**, you can find codes to implement GS model and Sufana model.
* In the part **result**, you can find our convergence test results and other analyses.
* In the part **docs**, you can find 2 pdf and 2 slides who explain the theoric idea of this project. The file named by mid will talk about the simulation of CIR and Wishart process, and the file named by final will talk about the application models of Wishart process.
* Some demonstration can be found in **.ipynb** files.

