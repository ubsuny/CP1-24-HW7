import numpy as np

def a_trap(y, d):
    """
    trap takes in y as an array of y values
    and d, the separation between each y-value
    to use the numpy trapezoid function to 
    return the integral
    """
    return np.trapezoid(y,dx=d)

def sec_derivative(func, x,dx):
    return np.gradient(np.gradient(func(x)),dx)

def adapt(func, bounds, d, sens):
    """
    adapt uses adaptive trapezoidal integration
    to integrate a function over boundaries.
    func must be a str defining the function
    to be integrated. May be:
    x^3+1
    exp(-1/x)
    cos(1/x)
    bounds is a list of length two which defines
    the lower and upper bounds of integration
    d defines the number of points between
    lower and upper bound to use with integration
    sens must be a number; it defines the 
    sensitivity the adapt function will have to 
    how the function changes. 
    """
    #The x is defined as a linspace which is used to define
    #the derivative of the function for each point x
    #between the bounds
    x=np.linspace(bounds[0], bounds[1], d+1)
    dx=x[1]-x[0]
    d2ydx2=sec_derivative(func,x,dx)

    loopx=enumerate(x)
    summer=0
    #a loop is run through x. Each second derivative is used to
    #define the number of indices of new_x, which is a
    #list defining a number of points inbetween 2 x values
    #then, trapezoidal integration is conducted over new_x
    #and each integration is summed with eachother to produce
    #the total integral.
    for count, val in loopx:
        if count!=len(x)-1:
            new_x=np.linspace(val, x[count+1], 2*(int(np.abs(sens*d2ydx2[count]))+1))
            new_y=func(new_x)
            summer+=a_trap(new_y, dx/((2*(int(np.abs(sens*d2ydx2[count]))+1))-1))
    return summer

def cubic(x):
    vals=[]
    for i in x:
        vals.append(i**3+1)
    return vals

def cosine(x):
    vals=[]
    for i in x:
        vals.append(np.cos(1/i))
    return vals

def exp(x):
    vals=[]
    for i in x:
        vals.append(np.exp(-1/i))
    return vals

