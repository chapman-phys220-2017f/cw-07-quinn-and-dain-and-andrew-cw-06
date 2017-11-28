
------------------
# PHYS220 CW 7

**Author(s):** **CHANGEME**

[![Build Status](https://travis-ci.org/chapman-phys220-2017f/cw-07-YOURNAME.svg?branch=master)](https://travis-ci.org/chapman-phys220-2017f/cw-07-YOURNAME)

## Specification

In this assignment, we will remind ourselves how Taylor's theorem for approximating functions works. This will use the derivative functionality you have already developed.

1. Copy the python module ```array_calculus.py``` from CW06 into this repository. You will use the derivative matrix that you defined to simplify the assignment.
1. Recall the Taylor expansion formula: $$f(a + y) = \sum_{n=0}^\infty \frac{y^n}{n!}\left[\frac{d^n}{dy^n}f(a+y)\right]_{y=0}\\ = f(a) + f'(a)y + f''(a)\frac{y^2}{2} + \cdots$$ This formula states that an analytic function $f$ can be approximated by a polynomial series in the region around a domain point $a$, with the polynomial weights determined only by the derivatives of $f$ at the single point $a$. The purpose of this assignment is to show how this formula really works in practice. By shifting the domain point to $x = a + y$, we can rewrite the formula as: $$f(x) = \sum_{n=0}^\infty \frac{(x-a)^n}{n!}\left[\frac{d^n}{dx^n}f(x)\right]_{x=a}\\ = f(a) + f'(a)(x-a) + f''(a)\frac{(x-a)^2}{2} + \cdots$$ This formulation makes it more clear that the function $f$ at any point $x$ in the domain may be approximated by derivatives at $a$ and the displacements $(x-a)$ away from the point $a$.
1. In a notebook ```taylor.ipynb``` explain the above formula using equations and your own words.
1. In a separate module ```taylor_approx.py``` import ```array_calculus```, and define a function ```taylor(x, f, i, n)``` that does the following. The parameter `x` should be an array of domain points. The parameter `f` should be an array of range points for a function $f(x)$. The integer `i` should be between 0 and the length of the domain, and will be used to index one point from the domain to "Taylor-expand around". The integer `n` should be positive, and will indicate how many terms to keep in the Taylor expansion sum. Implement this function so it returns two arrays `(x, fapprox)`, where `x` is the same as the input domain array, and `fapprox` is a new approximate function range obtained by applying the Taylor formula at the domain point `x[i]` contained at the index `i`, keeping only `n` terms of the expansion. (Hint: Use powers of the derivative matrix to easily compute derivatives.)
1. Write test functions in a test module ```test_taylor_approx.py``` that verify your implementation is correct.
1. In your notebook, demonstrate how this approximation works. That is, define a domain array `x` of $1000$ points between $[-5,5]$. Define range arrays for the following interesting functions: (1) $f(x) = \tanh(x)$, (2) $f(x) = x^2/10 + \sin(2x)/2$, (4) $f(x) = 1/x$ (Hint: use the special value `np.nan` to represent an undefined range value at $x=0$.) and (3) $f(x) = \Theta(x)$. (Recall the Heaviside Theta function $\Theta(x)$ is defined such that it is zero for $x<0$, $1$ for $x >0$ and $1/2$ for $x = 0$.) For each of these functions, plot the function and the Taylor approximations of the function for $n=0,1,2,3,4,5$ around a few interesting points. (Make a separate plot showing approximations for each point you choose to expand around, for each function. Draw a dot on the graph on the chosen point in each case to make it clear which point you are considering. Label your plots accordingly to make it clear what you are showing.)
1. Discuss your observations in your notebook. When is the Taylor expansion a good approximation? When does it have difficulty?

Pro-tip: using git to manage conflicts on Jupyter notebooks is a pain. I recommend delegating one person from your group to edit the notebook, to avoid merge conflicts.

## Assessment

Analyze in this section what you found useful about this assignment in your own words. Include any lingering questions or comments that you may have.

**CHANGEME**

## Honor Pledge

I pledge that all the work in this repository is my own with only the following exceptions:

* Content of starter files supplied by the instructor;
* Code borrowed from another source, documented with correct attribution in the code and summarized here.

Signed,

**YOURNAME**
