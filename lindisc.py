"""
prpy module lindisc.py
Jason Corso (jcorso@acm.org)

This module has been programmed to support teaching an introduction to
 pattern recognition course.

Contains implementations of linear discriminants.

This file is organized alphabetically by function name.

Some pointers on convention:
    1.  Generally, data is stored in an N x D matrix, where N is the number of samples and D is the dimension.
    2.  Class information is typically stored in a parallel N x 1 vector.
    3.  All plotting functions assume the data is 2 dimensions.

Please report bugs/fixes/enhancements to jcorso@acm.org when you find/make them.
"""

# local imports
import datatools

# global imports
import sys
import numpy as np
import matplotlib.pyplot as plt


#### Utility Functions First

def debugPlot(X,Y,a,b=0,h=None,Z=None,m=None):
    ''' Debug two-class plotting and draw the linear discriminant. 
    
        (X,Y) are the data set
        Z is the normalized data set

        a is the weight vector, b (if exists) is the bias (a'x + b is the classifier)
        m is the margin (if exists)

        h is the figure handle to draw into
    '''

    if h is None:
        h = plt.figure()
    else:
        plt.figure(h.number)

    plt.clf()

    plt.axis([-10,10,-10,10])
    plt.grid(True)
    plt.show()

    if Z is not None:
        B = (a*Z).sum(1) + b <= 0   
        for i in range(Y.shape[0]):
            if not B[i]:
                plt.plot(X[i,0],X[i,1],'go',hold=True,markersize=14.0, \
                        markerfacecolor=[1,1,1],markeredgecolor='g',markeredgewidth=1.5)

    for i in range(Y.shape[0]):
        if Y[i] == -1:
            plt.plot(X[i,0],X[i,1],datatools.kDraw[0],markersize=8.0,hold=True)
        else:
            plt.plot(X[i,0],X[i,1],datatools.kDraw[1],markersize=8.0,hold=True)

    plotWVector(a,b,m)

    return h

def normalize(X,Y):
    ''' Data normalization procedure: multiple data values by -1 if the class is -1. '''

    Xhat = X.copy()
    Xhat[Y==-1] *= -1;
    return Xhat

def plotWVector(a,b=None,m=None):
    ''' In the current axes, plot the weight vector. 
     
        a is the weight vector, b is the bias term
        m is the margin
    '''

    # O is some arbitrary point on the line, say where x=0
    O = np.zeros(2) 
    if b is not None:
        O[1] = -b/a[1]

    an = a / np.linalg.norm(a)
    V = O + 50*an
    U = O + -50*an

    plt.plot([O[0],U[0]],[O[1],U[1]],'b',linewidth=3.0)
    plt.plot([O[0],V[0]],[O[1],V[1]],'r',linewidth=3.0)

    anr = np.asarray([an[1],-an[0]])

    A = O + 50*anr;
    B = O + -50*anr;
    plt.plot([A[0],B[0]],[A[1],B[1]],'k',linewidth=3.0)

    if m is not None and np.isscalar(m):
        Ob = O - m*an
        Ab = Ob + 50*anr;
        Bb = Ob + -50*anr;
        plt.plot([Ab[0],Bb[0]],[Ab[1],Bb[1]],'k-',linewidth=1.0)

        Ob = O + m*an
        Ab = Ob + 50*anr;
        Bb = Ob + -50*anr;
        plt.plot([Ab[0],Bb[0]],[Ab[1],Bb[1]],'k-',linewidth=1.0)




#### A linear discriminant class
class LinDisc(object):
    '''
    A Linear Discriminant Classifier

    Bias included but can be disregarded.

    '''

    def __init__(self,w=None,b=None):
        self.w = w   # the weight vector 
        if b is None:
            self.b = 0
        else:
            self.b = b


    def classify(self,sample):
        ''' Classify the sample using the weight vector. '''
        return np.sign(self.w.dot(sample)+self.b)


#### Method Functions Next

def batchPerceptron(X,Y,a0,eta):
    ''' 
    Run the batch perceptron algorithm on (X,Y) to learn a linear discriminant. 
    
      Batch Perceptron on D (not normalized)
        for dimension d and n samples
      X is n by d 
      Y is n by 1 and is -1 or +1 for classes
      Assume (linear) separability.

        sum_{y in incorrect} -a'y

      for all x in D (column vector)

       a'x > 0

      no margin

      Basic assumption we have 2D points in a plane, for debugging and visualization.
    '''

    Z = normalize(X,Y)

    n = len(Y)

    t=0
    a = a0.copy()
    a = a / np.linalg.norm(a)

    h = debugPlot(X,Y,a,Z=Z)

    while(t<10):

        print 'iteration %d   %.4f  %.4f\t'%(t,a[0],a[1]),

        B = (a*Z).sum(1) <= 0    # note NumPy is auto-tiling a here...

        print '   %d of %d samples correct'%(sum(~B),n)

        # plot the current solution now
        debugPlot(X,Y,a,h=h,Z=Z)
        raw_input()

        if (sum(B) == 0):
            break
        
        a = a + (eta * Z[B].sum(0))
        a = a / np.linalg.norm(a)

        t += 1
        
    return a

def mse(X,Y,b):
    ''' 
    Run a least-squares estimation of the discriminant via the pseudo-inverse.

      X is n by d 
      Y is n by 1 and is -1 or +1 for classes

      Linear separability is not needed and a "best" answer will still be returned.

    '''

    Z = normalize(X,Y)
    n = len(Y)

    if np.isscalar(b):
        b = np.ones(n)*b

    a = np.linalg.pinv(Z).dot(b)
    a = a / np.linalg.norm(a)
    # could have also used linalg.lstsq(Z,b) 

    h = debugPlot(X,Y,a,Z=Z,m=b)

    return a

def ssPerceptron(X,Y,a0,eta):
    ''' 
    Run the single-sample perceptron algorithm on (X,Y) to learn a linear discriminant. 
    
      Batch Perceptron on D (not normalized)
        for dimension d and n samples
      X is n by d 
      Y is n by 1 and is -1 or +1 for classes
      Assume (linear) separability.

        sum_{y in incorrect} -a'y

      for all x in D (column vector)

       a'x > 0

      no margin

      Basic assumption we have 2D points in a plane, for debugging and visualization.
    '''

    Z = normalize(X,Y)
    n = len(Y)

    i=0 # sample index
    t=0 # overall iteration counter
    a = a0.copy()
    a = a / np.linalg.norm(a)

    h = debugPlot(X,Y,a,Z=Z)

    count=0
    while(count<n):

        count += 1

        val = a.dot(Z[i,:])

        print 'sample %d, val %0.3f, current count is %d of %d' % (i,val,count,n)
        if val <= 0:
            count = 0

            # train on it
            t += 1
            print 'correction iteration %d      %.4f  %.4f' % (t,a[0],a[1])

            debugPlot(X,Y,a,h=h,Z=Z)
            plt.plot(X[i,0],X[i,1],'yo',hold=True,markersize=14.0)

            a = a + (eta * Z[i,:].squeeze())
            a = a / np.linalg.norm(a)

            s = raw_input()
            if s == 'q':  # this pauses until getting a carriage return
                break
            #plt.pause(0.1)

        i = (i+1)%n     

    debugPlot(X,Y,a,h=h,Z=Z)

    return a


def ssRelaxation(X,Y,a0,eta,b):
    '''
    Run the single-sample relaxation with margin procedure.

      Single-Sample Relaxation with the margin on D (not normalized)
       for dimension d and n samples
      D is n by d + 1 where the last column of D is
      Y is n by 1 and is -1 or +1 for classes
      Assume separability with the margin

        1/2 sum_{y in incorrect} (a'y - b)^2 / ||y||

      for all x in D (column vector)

    '''

    Z = normalize(X,Y)

    n = len(Y)

    i=0 # sample index
    t=0 # overall iteration counter
    a = a0.copy()
    a = a / np.linalg.norm(a)

    h = debugPlot(X,Y,a,Z=Z,m=b)

    count=0
    while(count<n):

        count += 1

        val = a.dot(Z[i,:])

        print 'sample %d, val %0.3f, current count is %d of %d' % (i,val,count,n)
        if val <= b:
            count = 0

            # train on it
            t += 1
            print 'correction iteration %d      %.4f  %.4f' % (t,a[0],a[1])

            debugPlot(X,Y,a,h=h,Z=Z,m=b)
            plt.plot(X[i,0],X[i,1],'yo',hold=True,markersize=14.0)

            dist = (b-val) / np.power(np.linalg.norm(Z[i,:].squeeze()),2)

            a = a + (eta * dist * Z[i,:].squeeze())
            a = a / np.linalg.norm(a)

            s = raw_input()
            if s == 'q':  # this pauses until getting a carriage return
                break
            #plt.pause(0.1)

        i = (i+1)%n     

    debugPlot(X,Y,a,h=h,Z=Z,m=b)

    return a



