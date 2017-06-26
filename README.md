# Support Vector Machines for face recognition:

Face recognition is a learning problem that has recently received a lot of attention. Support Vector Machines (SVM) are becoming very popular in the machine learning community as a technique for tackling high-dimensional problems. We implement the SVM algorithm as a face recognition tool. 

The experimental dataset can be found at "http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html". 

 The code is  written in Python, and for solving the QP, it  uses CVXOPT module to solve the problem.

For running the code you just need to set variable ‘path’ to the path of the dataset. 

We implement the SVM with soft margin, and we apply different combination of LDA, PCA, and SVM to get the better results. We implement Linear Kernel, 2nd, 4th, and 8th order of Polynomial kernel.  

For selecting your desire kernel, you just to change variable ‘kernel_type’ to ‘L’, P2, P4, or P8. 



