Numerical simulation of Inhomogeneous Geometric Brownian Motion (IGBM)

dy_t = a(b-y_t) dt + sigma y_t dW_t.


This code reproduces the results of the paper "An optimal polynomial approximation of Brownian motion" by James Foster, Terry Lyons and Harald Oberhauser.

In addition, the numerical methods and example are outlined in the document polynomial_presentation.pdf.

The source file igbm.cpp only requires headers from the C++ standard library.

The text file igbm_simulation.txt displays the output of the code. (In this case, the code was run on a laptop computer)


License

The content of polynomial_presentation.pdf is licensed under the Creative Commons Attribution 4.0 International license, and igbm.cpp is licensed under the MIT license.