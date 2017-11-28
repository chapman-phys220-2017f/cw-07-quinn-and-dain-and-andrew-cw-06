#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import array_calc as ac


def taylor(x,f,i,n):
    """taylor(x,f,i,n)
    Inputs:
      -x: array of domain points
      -f: array of range points for a function f(x)
      -i: integer, between 0 and len(x), such that x[i] will be the point expanded around
      -n: positive integer, number of terms kept in the approximation
    Returns: (x, fapprox)
      -x: same array of domain points as inputted
      -fapprox: array of the taylor approximated function at point x[i]"""
    length = len(x)
    mat = ac.derivative(x[0],x[length-1],length)
    fapprox = np.zeros(length)
    def mat_generator():
        n_mat = np.eye(length)
        while True:
            n_mat = np.dot(n_mat,mat)
            yield n_mat
    g = mat_generator()
    k = np.arange(length)
    a = x[i]
    fapprox[k] = f[i]
    for j in range(1,n):
        n_mat = next(g)
        fapprox += (np.dot(n_mat,f)[i]*((x-a)**j)/(np.math.factorial(j)))
    return (x,fapprox)

def taylor(x,f,i,n):
    """Parameter x is an array of domain points. 
    The parameter f should be an array of range points for a function $f(x)$. 
    The integer i should be between 0 and the length of the domain, and will be used to index one point from the domain to "Taylor-expand around". 
    The integer n should be positive, and will indicate how many terms to keep in the Taylor expansion sum. 
    This function returns two arrays (x, fapprox), where x is the same as the input domain array, and fapprox is a new approximate function range 
    obtained by applying the Taylor formula at the domain point x[i] contained at the index i, keeping only n terms of the expansion."""
    #total = 0
    #dx = x[i] - x[i-1]
    #for j in range(n):
    #    mat = np.linalg.matrix_power(np.matrix(ac.derivative(x[0],x[-1],n)),j) @ f
    #    total = total + (mat[i-1]*(dx**j))
    def tay(e):
        total = 0
        for j in range(n):
            mat = np.linalg.matrix_power(np.matrix(ac.derivative(x[0],x[-1],n)),j) @ f
            total = total + (mat[i]*((x[e]-x[i])**j)/np.factorial(j))
        return total
    fx = np.vectorize(tay)
    return (x,fx(x))
