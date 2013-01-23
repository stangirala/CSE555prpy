"""
prpy module datatools.py
Jason Corso (jcorso@acm.org)

This module has been programmed to support teaching an introduction to
 pattern recognition course.

Contains tools for handling data (sets, samples, and so on) and evaluating it.

This file is organized alphabetically by function name.

Some pointers on convention:
    1.  Generally, data is stored in an N x D matrix, where N is the number of samples and D is the dimension.
    2.  Class information is typically stored in a parallel N x 1 vector.
    3.  All plotting functions assume the data is 2 dimensions.

Please report bugs/fixes/enhancements to jcorso@acm.org when you find/make them.
"""


import numpy as np
import matplotlib.pyplot as plt

# Some constants that are used in the script
kDraw = [ 'bo','ro','go','b+','r+','g+','bx','rx','gx' ]


def accuracy(X,Ytrue,classifier):
    '''
    Function to score the accuracy on data set X,Ytrue with known labels.

    The classifier is assumed to expose a .classify function that will take a row-vector
    and output a classification that can be corresponded to the Ytrue entry.
    '''
    a = 0;
    for i in range(X.shape[0]):
        if (Ytrue[i] == classifier.classify(X[i,:])):
            a += 1;

    return a/np.double(len(Ytrue))

def classifySet(X,classifier):
    '''
    Function to classify an entire set of data and return a classification vector Y.

    The classifier is assumed to expose a .classify function that will take a row-vector
    and output a classification that can be corresponded to the Ytrue entry.
    '''
    Y = np.zeros(X.shape[0],dtype=np.int16);
    for i in range(X.shape[0]):
        Y[i] = classifier.classify(X[i,:])
    return Y



def genData_gaussian(mu,sigma,n):
    '''  
    Generate a numpy array that is sampled from a Gaussian with parameters mu and sigma.

    The array returned is n by d where d is the dimension of mu.

    The implementation just uses the numpy.random.multivariate_normal to do the 
    work. The implementation does not seed the random number generator within 
    numpy, which should be done outside of this script if necessary.

    '''

    return np.random.multivariate_normal(mu,sigma,n)

def genData_interactive(numClasses,numClicks):
    '''
    Generate a 2D data set interactively by letting the user click on the axes to distribute the points.  Each click generates 20 points randomly sampled from an isotropic Gaussian.  Multiple classes are possible.  numClicks it the number of clicks per class.

    Data set array X is returned (numClasses*numClicks x 2) and Y Classes vector
    '''
    kPer = 20
    kSigma = [[0.3,0.0],[0.0,0.3]]

    plt.figure() 
    plt.axis([-10,10,-10,10])
    plt.show()

    X = np.zeros([kPer*numClicks*numClasses,2])
    Y = np.zeros([kPer*numClicks*numClasses],dtype=np.int16)

    for c in range(numClasses):
        for i in range(numClicks):
            plt.title('Sampling Clicks For Class %d/%d, Click %d/%d'%(c,numClasses,i,numClicks))

            x = np.squeeze(plt.ginput(1))
            xsamples = genData_gaussian(x,kSigma,kPer) 

            current = c*kPer*numClicks + i*kPer
            X[current:current+kPer,:] = xsamples
            Y[current:current+kPer  ] = c*np.ones([kPer],dtype=np.int16)

            plt.plot(xsamples[:,0],xsamples[:,1],kDraw[c%len(kDraw)],hold=True)

    plt.title('Done Sampling Data')

    return (X,Y)


def plotData(X,Y):
    ''' A general plotting function on 2D data set (X,Y).  Can be multiclass. '''

    plt.figure()
    plt.axis('equal')

    # will need to handle the case that the data is two-class and -1,1 or m-class and 0,1,...
    if min(Y) == -1:
        for i in range(Y.shape[0]):
            if Y[i] == -1:
                plt.plot(X[i,0],X[i,1],kDraw[0],hold=True)
            else:
                plt.plot(X[i,0],X[i,1],kDraw[1],hold=True)
    else:
        for i in range(Y.shape[0]):
            plt.plot(X[i,0],X[i,1],kDraw[Y[i]%len(kDraw)],hold=True)
    plt.show()


def plotData_one(X):
    ''' Plot the data set X. '''

    plt.plot(X[:,0],X[:,1],'b+')
    plt.axis('equal')
    plt.show()


def plotData_twoClass(X1,X2):
    ''' Plot two data sets in the same axes (D1 is blue and D2 is red). '''
    
    plt.plot(X1[:,0],X1[:,1],'b+')
    plt.plot(X2[:,0],X2[:,1],'ro')
    plt.axis('equal')
    plt.show()
    
    
def testGrid(classifier):
    ''' Evaluate the classifier over a uniformly spaced grid ... '''

    A,B = np.mgrid[-10:10:0.5,-10:10:0.5]
    X   = np.asarray([A.ravel(),B.ravel()]).T

    num = X.shape[0]
    Y   = np.zeros(num,dtype=np.int16)

    for i in range(num):
        Y[i] = classifier.classify(X[i,:])

    plotData(X,Y)

    return (X,Y)
