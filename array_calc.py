#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def derivative(a,b,n):
    """derivative(a,b,n)
    Function to create the derivative matrix.  Takes 3 inputs:
        - a = float, starting point of the domain
        - b = float, end point of domain
        - n = number of points created in the domain
    Output: an 'n x n' 2-dimensional numpy array that, when matrix multiplied by a 1-D array,
    returns the approximate derivative at each point."""
    t = np.linspace(a,b,n)
    dx = t[1]-t[0]
    empty = np.zeros(n, dtype = 'float64')
    a,b = np.meshgrid(empty, empty)
    A = a + b
    #Forward Difference
    A[0,0] = -1/dx
    A[0,1] = 1/dx
    #Symmetric Difference
    index = np.arange(1,n-1)
    A[index, index-1] = -1/(2*dx)
    A[index, index+1] = 1/(2*dx)
    #Backward Difference
    A[n-1,n-2] = -1/dx
    A[n-1,n-1] = 1/dx
    return A

def second_derivative(a,b,n):
    """second_derivative(a,b,n)
    Function to create the second derivative matrix.  Takes 3 inputs:
        - a = float, starting point of the domain
        - b = float, end point of domain
        - n = number of points created in the domain
    Output: an 'n x n' 2-dimensional numpy array that, when matrix multiplied by a 1-D array,
    returns the approximate second derivative at each point."""
    t = np.linspace(a,b,n)
    dx = t[1]-t[0]
    empty = np.zeros(n, dtype = 'float64')
    a,b = np.meshgrid(empty, empty)
    A = a + b
    #Forward Difference
    A[0,0] = 1
    A[0,1] = -2
    A[0,2] = 1
    #Symmetric/Forward Difference
    A[1,0] = 2
    A[1,1] = -3
    A[1,3] = 1
    #Symmetric Difference
    index = np.arange(2,n-2)
    A[index, index-2] = 1/2
    A[index, index] = -1
    A[index, index+2] = 1/2
    #Symmetric/Backward Difference
    A[n-2,n-4] = 1
    A[n-2,n-2] = -3
    A[n-2,n-1] = 2
    #Backward Difference
    A[n-1,n-3] = 1
    A[n-1,n-2] = -2
    A[n-1,n-1] = 1
    return A*(1/(2*dx**2))

def f(a,b,n):
    """Returns an array of the square of the input (a float)"""
    t = np.linspace(a,b,n)
    return t**2

def s(a,b,n):
    """Returns an array of the sine of the input (float)"""
    t = np.linspace(a,b,n)
    sin = np.vectorize(np.sin)
    sin = sin(t)
    return sin

def g(a,b,n):
    """Returns an array of the gaussian function"""
    def gauss(x):
        """Returns the Gaussian function dependent on the input float"""
        c = 1/(np.sqrt(2*np.pi))
        gauss = c*np.exp(-x**2/2)
        return gauss
    t = np.linspace(a,b,n)
    gs = np.vectorize(gauss)
    gs = gs(t)
    return gs

def plot_function(a,b,n,f,string):
    """plot_function(a,b,n,f,string)"""
    t = np.linspace(a,b,n)
    deriv = np.dot(derivative(a,b,n),f)
    sec_deriv = np.dot(second_derivative(a,b,n),f)
    plt.plot(t,f,'b', label=string)
    plt.plot(t,deriv,'r', label='Derivative')
    plt.plot(t,sec_deriv,'g', label='Second Derivative')
    plt.title(string)
    plt.legend()
    plt.show()
