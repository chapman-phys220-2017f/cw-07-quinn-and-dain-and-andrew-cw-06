#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import nose
import taylor_approx as ta
import array_calc as ac

"""Test function for our taylor approximation"""

def test_taylor_approx():
    """Tests our function taylor(x,f,i,n) to approximate the Gaussian function around
    some point arbitrarily chosen."""
    a,b,n = 0,3,1000
    t = np.linspace(a,b,n)
    desired = ac.g(a,b,n)
    # Values obtained from taylor approximation
    actual = ta.taylor(t,desired,501,10)
    # Debug message
    print("We expected this: ",desired[500:503]," but the result was: ",actual[1][500:503])
    # Testing accuracy
    for k in range(500,502):
        nose.tools.assert_almost_equal(desired[k],actual[1][k],4)
