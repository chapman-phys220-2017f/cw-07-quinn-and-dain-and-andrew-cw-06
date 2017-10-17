#!/usr/bin/env python3

import array_calculus as ac
import numpy as np

def taylor(x,f,i,n):
    """The parameter x should be an array of domain points. 
    The parameter f should be an array of range points for a function $f(x)$. 
    The integer i should be between 0 and the length of the domain, and will be used to index one point from the domain to "Taylor-expand around". 
    The integer n should be positive, and will indicate how many terms to keep in the Taylor expansion sum. 
    This function returns two arrays (x, fapprox), where x is the same as the input domain array, and fapprox is a new approximate function range 
    obtained by applying the Taylor formula at the domain point x[i] contained at the index i, keeping only n terms of the expansion."""
    total = 0
    fapprox = []
    dx = x[i] - x[i-1]
    for j in range(n):
        mat = np.linalg.matrix_power(np.matrix(ac.derivative(x[0],x[-1],n)),j) @ f
        total = total + (mat[i-1]*(dx**j))
    