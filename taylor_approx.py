#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import array_calc as ac

"""Code module containing one function 'taylor' that approximates a function using Taylor expansion."""

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