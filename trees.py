"""
prpy module trees.py
Jason Corso (jcorso@acm.org)

This module has been programmed to support teaching an introduction to
 pattern recognition course.

Contains tree classifiers

"""

# local imports
import datatools

# global imports
import numpy as np

kDT_MaxDepth = 5    # maximum depth we'll grow a decision tree
kDT_LenRanges = 100 # how many thresholds to consider when selecting AxisAligned query at a node
kVerbose = True


class DTNode(object):
    '''
    A Decision Tree Node class.

    Links to children via reference list.
    '''

    def __init__(self,q=None,left=None,right=None):
        self.q        = q
        self.child    = [left,right]

    def isLeaf(self):
        return False

    def walk(self,sample):
        if self.q.query(sample)==0:
            return self.child[0]
        else:
            return self.child[1]
        

class DTLeafNode(DTNode):
    ''' 
    A Leaf Node of the decision tree that stores an empirical density.

    '''

    def __init__(self,density=None):
        self.density = density
        if (self.density):
            self.computeMode_()

    def computeDensity(self,X,Y):
        ''' Compute and store the empirical density at this leaf. '''
        U = np.unique(Y)

        self.density = np.zeros(U.max()+1)

        for c in U:
            yc = np.nonzero(Y==c)[0]   # like matlab's find()
            if (len(yc)==0):
                continue
            self.density[c] = len(yc)/np.double(len(Y))

        self.computeMode_()


    def computeMode_(self):
        ''' Compute and store the mode of the empirical density at this leaf. '''
        m = 0.0
        self.mode = None
        for i in range(len(self.density)):
            if (self.density[i] > m):
                m = self.density[i]
                self.mode = i

        if kVerbose:
            print "Density is " , self.density
            print "Mode of the leaf is %d\n" % self.mode

    def isLeaf(self):
        return True

    def walk(self,sample):
        return self


class DTQuery(object):
    ''' An empty parent class to store a query for the decision tree. '''
    pass


class DTQuery_AxisAligned(DTQuery):
    ''' An axis-aligned query (basically thresholds on one coordinate in the value array. '''

    def __init__(self,query_index=None,tau=None):
        self.qi  = query_index
        self.tau = tau

    def __repr__(self):
        return "Query_AxisAligned:  index: %d, tau %f\n" % (self.qi,self.tau)
    
    def query(self,value):
        ''' 
        A simple query response tester on the value (a vector) index query_index.

        If the value is less then tau, a 0 is returned and if it is greater than tau a 1 is returned.
        '''
        if value[self.qi] < self.tau:
            return 0
        else:
            return 1


class DTree(object):
    '''
    A Decision Tree class.

    '''

    def __init__(self,root=None):
        self.root = root


    def classify(self,sample):
        ''' Classify the sample using the tree. '''
        leaf = self.findLeaf(sample)
        return leaf.mode

    def findLeaf(self,sample): 
        ''' Find the leaf node in the tree for a given sample. '''
        node = self.root;
        while (node.isLeaf() is not True):
            node = node.walk(sample)
        return node

    def probability(self,sample):
        ''' Return the empirical distribution over the classes for the sample. '''
        leaf = self.findLeaf(sample)
        return leaf.density



def impurity_entropy(Y):
    ''' Calculate the entropy impurity for a data set (Y). '''

    U = np.unique(Y)

    H = 0.0;

    for c in U:
        yc = np.nonzero(Y==c)[0]   # like matlab's find()

        if (len(yc)==0):
            continue

        cportion = len(yc)/np.double(len(Y))
        H += cportion * np.log(cportion)

    return -1.0 * H


def trainDTree_prepRanges(X):
    ''' 
    Prepare a ranges list for training in the trainDTree_AxisAligned.
    
    The ranges list just basically goes through the data-range of each dimension in X and 
    gathers a set of possible thresholds for it.
    '''

    R = []

    for d in range(X.shape[1]):
        dmin = np.min(X[:,d])
        dmax = np.max(X[:,d])
        dstep = np.abs((dmax-dmin)/np.double(kDT_LenRanges))
        
        if (dstep < 0.00001):
            R.append(np.zeros(kDT_LenRanges))
        else:
            R.append(np.arange(dmin,dmax,dstep))

    return R


def trainDTree_AxisAligned(X,Y,impurity,depth=0):
    '''
    Train a decision tree using data set (X,Y).

    Convention on (X,Y) is that each row is a sample in the data set, X has the 
    values of the data and Y has the class label.

    impurity is a function that takes X,Y and returns the impurity of the data set

    ***Specific to an axis-aligned query case, which is a fairly general case...
    ***Code flow for clarity rather than speed...
    '''

    # could assume there is some impurity coming in, but just recompute it
    I = impurity(Y)
    if depth>kDT_MaxDepth or I==0:
        ''' We're at a leaf-node and need to return as such. '''
        leaf = DTLeafNode()
        leaf.computeDensity(X,Y)
        return leaf

    ranges = trainDTree_prepRanges(X)

    dim = len(ranges)
    assert(dim == X.shape[1])
    num = X.shape[0]

    bestIg = 0

    Q = DTQuery_AxisAligned()
    bestQ = DTQuery_AxisAligned()
    bestL = None  # store the splits for simplicity
    bestR = None

    # cycle through the data indices and possible thresholds to find the best one (the arg max)
    for qi in range(dim):
        Q.qi = qi
        for t in ranges[qi]:
            Q.tau = t

            # calculate the splits
            L = []
            R = []
            for i in range(num):
                if Q.query(X[i,:])==0:
                    L.append(i)
                else:
                    R.append(i)

            # calculate the impurity gradient
            Lportion = len(L) / np.double(num)
            Limpurity = impurity(Y[L])
            Rportion = len(R) / np.double(num)
            Rimpurity = impurity(Y[R])
    
            Ig = I - Lportion*Limpurity - Rportion*Rimpurity
            
            if (Ig > bestIg):
                bestIg    = Ig
                bestQ.qi  = qi
                bestQ.tau = t
                bestL     = L
                bestR     = R
            
    # If we cannot find a split at this node, just turn it into a leaf
    # BUGFIX 2/17/12 jcorso
    if (bestL is None) or (bestR is None):
        leaf = DTLeafNode()
        leaf.computeDensity(X,Y)
        return leaf

    # bestQ is the arg max_queries of impurity gradient
    leftChild  = trainDTree_AxisAligned(X[bestL,:],Y[bestL],impurity,depth+1)
    rightChild = trainDTree_AxisAligned(X[bestR,:],Y[bestR],impurity,depth+1)

    if (depth>0):
        return DTNode(bestQ,leftChild,rightChild)
    else:
        return DTree(DTNode(bestQ,leftChild,rightChild))



