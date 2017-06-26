from scipy import misc
import os, copy, random, math
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from collections import Counter
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# Please set 'path' to the address of dataset
path = "F:\\OneDrive\\Mohsen\\6363\\HW2\\6363\\final\\att_faces\\"
# put your desire Kernel in here
kernel_types = ['L']
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
flag_1 = 0
flag_2 = 0
training_class = []
testing_class = []

class a_and_b:
    def __init__(self):
        self.alpha = None
        self.b = None
        self.yy = None

#--------------------reading images from HD-----------------
def open_images(path,siz = None):
    print "Reading images....."
    if siz is not None:
        print "   Images will be resized by ", siz
    #path = "F:\\OneDrive\\Mohsen\\6363\\HW2\\6363\\final\\att_faces\\"
    folder_list = os.listdir(path)
    flag_1 = 0
    flag_2 = 0
    training_class = []
    testing_class = []
    rnd = [2,4,7,10,9]
    for folder in folder_list:
        file_path =path + folder
        file_list = os.listdir(file_path)
        for i in range(1,11):
            im = misc.imread(file_path + '\\' + str(i) + '.pgm')
            if siz is not None:
                im = misc.imresize(im, siz)
            shape = im.shape
            im = im.reshape((im.shape[0]*im.shape[1], 1))
            if i in rnd:#-------------Training matrix-----------------
                if flag_1 == 0:
                    X = np.copy(im)
                    flag_1 = 1
                    training_class.append(folder)
                else:
                    X  = np.hstack((X, np.copy(im)))
                    training_class.append(folder)
            else:#--------testing matrix--------------
                if flag_2 == 0:
                    T = np.copy(im)
                    flag_2 = 1
                    testing_class.append(folder)
                else:
                    T  = np.hstack((T, np.copy(im)))
                    testing_class.append(folder)
    print '   We read {0} images as training and {1} images as testing and saved them in matrixes in  ({1} x {2})'.format(X.shape[1],T.shape[1], X.shape[0],X.shape[1])
    #display_me(X, X.shape[1],shape, "Original faces")
    #X1 = X.astype(np.float, copy = True)
    #T1 = T.astype(np.float, copy = True)
    Xclasses = []
    Tclasses =[]
    for c in folder_list:
        s = np.empty([X.shape[0],0])
        for clm in range(X.shape[1]):
            if (c == training_class[clm]):
                s = np.hstack((s, X[:,clm].reshape((X.shape[0],1))))
        Xclasses.append(s)
        st = np.empty([T.shape[0],0])
        for clm in range(T.shape[1]):
            if (c == testing_class[clm]):
                st = np.hstack((st, T[:,clm].reshape((T.shape[0],1))))
        Tclasses.append(st)

    return X,training_class,T,testing_class, shape, Xclasses, Tclasses

# eigenvalues and eigenvectors
def eigen(X, shape= None):
    print "Eigenvalues and eigenvectors finder: Running...."
    im_mean = X.mean(axis = 1)# mean face
    if shape is not None:
        dis_im_mean = im_mean.reshape(shape) # reshape to display it
    # subtracting all faces from mean
    reshaped_mean = im_mean.reshape((im_mean.shape[0],1))
    print "   Subtracting mean face from matrix X ..."
    X_ = X - reshaped_mean
    # transpose of matrix------------------------
    X_T = X_.transpose()
    print "   Transposing matrix X, the shape will be {0}x{1} ".format(X_T.shape[0],X_T.shape[1])

    #---muliplying A^T and A-----------------
    mult = np.dot(X_T,X_)/float(X.shape[1] - 1)

    print "   Computing coveriance matrix (X^T * X)/({0} - 1), the shape would be {0}x{1} ".format(X.shape[1],mult.shape[0],mult.shape[1])

    #eigenvalues and eigenvectors---------
    e_values, e_vectors = np.linalg.eigh(mult)
    #idx = e_values.argsort()[::-1]
    idx = np.argsort(-e_values)
    e_values = e_values[idx]
    e_vectors = e_vectors[:,idx]
    print "   Computing eigenvalues and eigenvectors and sort them descendingly, eigenvectors is {0}x{1} ".format(e_vectors.shape[0],e_vectors.shape[1])
    #---------------------------------------------
    e_vectors_ = np.zeros((X.shape[0],e_vectors.shape[0]))# an eigen vectors matrix N by M ---eigenvectors of XX^T
    for i in range(e_vectors.shape[1]):
        e_vectors_[:,i] = np.dot(X_, e_vectors[:,i])
        e_vectors_[:,i] = e_vectors_[:,i]/np.linalg.norm(e_vectors_[:,i])

    print "   Computing X * Eigenvectors, eigenvectors is {0}x{1} and normalizing them ".format(e_vectors_.shape[0],e_vectors_.shape[1])
    #----------------------------------------------------------

    return e_values, e_vectors_
#Starting---------------------------------------------


def get_Sw(Xclasses, st):
    sz = st
    Sw = np.zeros((sz,sz))
    mean_tot = np.zeros((sz,1))
    mean_list = []
    for c in Xclasses:
        m = c.mean(axis = 1).reshape((-1,1))
        mean_list.append(m)
        mean_tot = mean_tot + m
        for cl in range(c.shape[1]):
            Sw = Sw + np.dot((c[:,cl].reshape((-1,1)) - m), ((c[:,cl].reshape((-1,1)) - m)).T)
    mean_tot =mean_tot/float(len(Xclasses))
    Sb = np.zeros((sz,sz))
    for m in mean_list:
        Sb = Sb + (np.dot((m - mean_tot),(m - mean_tot).T)*float(5))

    return Sw, Sb

# Here we compute the w transforer matrix for LDA
def LDA_w(Xclasses, shape, N = None):
    print "LDA: Running...."
    Sw,Sb = get_Sw(Xclasses,shape)
    print "   Sw's shape:", Sw.shape," , Sb's shape:  ", Sb.shape
    Sw_inv = np.linalg.inv(Sw)
    e_values,e_vectors = np.linalg.eigh(np.dot(Sw_inv,Sb))
    print "   number of eigenvalues:  ", len(e_values)
    idx = np.argsort(-e_values)
    e_values = e_values[idx]
    e_vectors = e_vectors[:,idx]
    d = len(Xclasses)
    if N is not None:
        d = N
    w = e_vectors[:,0:(d - 1)].copy()
    w = w.transpose()
    print "   w.T returned..."
    return w


def test_func(Y, tr_class, TY,tst_class):
    cntr = 0
    for i in range(TY.shape[1]):# i counts the testing
        dis = []
        for j in range(Y.shape[1]):# J counts the trianing
            dis.append(np.linalg.norm(Y[:,j] - TY[:,i]))
        inx_tr = dis.index(min(dis))
        if tr_class[inx_tr] == tst_class[i]:
            cntr = cntr + 1
    print "PCA accuracy rate = {}".format((float(cntr)/float(len(tst_class)))*100.0)
    return (float(cntr)/float(len(tst_class)))* 100.0

def kernel_func(x1,x2, type_):
    if ((type_ == "Polynomial4" or type_ == 'P4')):
        return math.pow((float(np.dot(x1,x2)) + 1.0), 4.0)
    elif ((type_ == "Polynomial8" or type_ == 'P8')):
        return math.pow((float(np.dot(x1,x2)) + 1.0), 8.0)
    elif ((type_ == "Polynomial2" or type_ == 'P2')):
        return math.pow((float(np.dot(x1,x2)) + 1.0), 2.0)
    else:
         return float(np.dot(x1,x2))



def get_P(K,y):
    p = matrix(np.outer(y,y) * K)
    return p

def kernel(x,type_):
    n = x.shape[1]
    K = matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel_func(x[:,i],x[:,j], type_)
    return K

def f_x(j,x,k,y,b_, a, const):
    n = x.shape[1]
    for i in range(n):
        if a[i] < const:
            a[i] = 0.0
    f = 0
    for i in range(n):
        f += y[i]*a[i]*k[j,i]
    f = f + b_
    return f


def f_x_testing(t,x,y,b_, a, const, type_):
    n = x.shape[1]
    a_tmp = []
    for i in range(n):
        if a[i] < const:
            a_tmp.append(0.0)
        else:
            a_tmp.append(a[i])
    f = 0
    for i in range(n):
        f += y[i]*a_tmp[i]*kernel_func(x[:,i],t, type_)
    f = f + b_
    return f

def get_b(K,y,a,const):
    sv = a > const
    ind = np.arange(len(a))[sv]
    #----------------compute 'b' ----------------------------------------
    sprt_y = []
    for i in ind:
       sprt_y.append(y[i])
    a_ = a[sv]
    #print len(a_)
    sprt_v = x[:,sv].copy()
    b_ = 0
    for i in range(len(a_)):
        b_ += sprt_y[i]
        for j in range(len(a_)):
            b_ -= a_[j] * sprt_y[j] * K[ind[i],ind[j]]
    b_ /= len(a_)
    return b_

def get_w_and_b(x,training_class,c,const, type_):
    sz = x.shape # dxn
    d = sz[0]
    n = sz[1]
    r = 0.1
    classes = Counter(training_class)
    classes = classes.keys()
    alpha_b = {}
    t1 = []
    t2 = []
    print sz
    #classes = ["s1","s30","s12", "s25"]
    #classes = classes[0:20]
    for cl in classes:
        y =[]
        for t in training_class:
            if t == cl:
                y.append(1.0)
            else:
                y.append(-1.0)
        #-----------------------------------------
        #y =[1.0,1.0,-1.0,-1.0,-1.0]
        #id1 = np.identity(n)
        '''id2 = -1.0 * np.identity(n)
        tmp = np.zeros((2 * n, 1), dtype= float)
        for i in range(n):
            tmp[i,0] = c
        tmp = np.zeros(( n, 1), dtype= float)
        q = matrix((-1.0)*np.ones((n,1)), tc = 'd')
        P = get_P(x, y, n, d)
        A = matrix(np.array(y).reshape((1,n)), tc = 'd')
        b = matrix(np.zeros((1,1)), tc = 'd')
        G = matrix(id2, tc = 'd')
        h = matrix(tmp.copy(), tc = 'd')
        print 'Here'
        sol = solvers.qp(P,q,G,h,A,b)
        my_alpha = sol['x']'''
        #-------------------------------------------------
        K = kernel(x, type_)
        #K = kernel_Polynomial(x,2.0)
        q = matrix(np.ones(n) * -1.0)
        P = get_P(K,y)
        A = matrix(y, (1,n))
        b = matrix(0.0)
        #G = matrix(np.diag(np.ones(n) * -1.0))
        #h = matrix(np.zeros(n))
        #----------------------------------------
        tmp1 = np.diag(np.ones(n) * -1.0)
        tmp2 = np.identity(n)
        G = matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n)
        tmp2 = np.ones(n) * c
        h = matrix(np.hstack((tmp1, tmp2)))

        solution = solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])
        b_ = get_b(K,y,a,const)
        alpha_b[cl] = a_and_b()
        alpha_b[cl].alpha = a.copy()
        alpha_b[cl].b = b_.copy()
        alpha_b[cl].yy = list(y)
        test = []
        for i in range(n):
            test.append(f_x(i,x,K,y,b_, a, const))
        tmp = 0
        for i in test:
            if (i >= 0.0):
                tmp +=1
        t1.append(tmp)
        tmp = 0
        for i in range(a.shape[0]):
            if (a[i] >= const):
                tmp += 1
        t2.append(tmp)
    print t1
    tmp = 0
    for i in t1:
        if (i == 5):
            tmp += 1
    print tmp, "/", len(t1), "  c = ",c, "cons = ", const
    print t2
    return alpha_b

def testing_phase(x,T,testing_class, all, const, type_):
    print "starting testing phase"
    cnt = 0
    for i in range(len(testing_class)):
        tmp = []
        tmp_class = []
        t_ = T[:,i]
        for cl in all:
            tmp.append(f_x_testing(T[:,i],x,all[cl].yy,all[cl].b, all[cl].alpha, const, type_))
            tmp_class.append(cl)
        '''tmp_ =[]
        tmp_class_ = []
        for j in range(len(tmp)):
            if (tmp[j] >=0.0):
                tmp_.append(tmp[j])
                tmp_class_.append(tmp_class[j])
        if ((len(tmp_) != 0)):'''
        indx = tmp.index(max(tmp))
        if (tmp_class[indx] == testing_class[i]):
            cnt += 1
    print 'Accuracy Rate:', (float(cnt)/float(len(testing_class)))*100.0
    return (float(cnt)/float(len(testing_class)))*100.0

def get_classes(X,training_class,folder_list):
    Xclasses = []
    for c in folder_list:
        s = np.empty([X.shape[0],0])
        for clm in range(X.shape[1]):
            if (c == training_class[clm]):
                s = np.hstack((s, X[:,clm].reshape((X.shape[0],1))))
        Xclasses.append(s)
    return Xclasses



folder_list = os.listdir(path)
X,training_class,T,testing_class, shape, Xclasses, Tclasses = open_images(path)
X = X.astype(np.float, copy = True)
T = T.astype(np.float, copy = True)
#Get matrix W from LDA
"""w = LDA_w(Xclasses, shape)
print "w's shape: ", w.shape
#The training  and testing set go to the new space by LDA
x = np.dot(w,X)
T = np.dot(w,T)
test_func(x,training_class, x, training_class)
print "X's shape after LDA: ", x.shape
"""
"""thr = 100
e_values, e_vectors = eigen(X,shape)
im_mean = X.mean(axis = 1)# mean face
reshaped_mean = im_mean.reshape((im_mean.shape[0],1))
X_ = X - reshaped_mean
p = e_vectors.transpose()
p = p[0:thr,:].copy()
X = np.dot(p, X_)
T_ = T - (T.mean(axis = 1)).reshape((-1,1))
T = np.dot(p, T_)
'''
Xclasses = get_classes(X,training_class,folder_list)
w = LDA_w(Xclasses, thr)
X = np.dot(w,X)
T = np.dot(w,T)'''

X = X.astype(np.float, copy = True)
T = T.astype(np.float, copy = True)"""
#---------------------------------------------------------
x = X.copy()
#x = np.array([(1.0,2.0,3.0,4.0,5.0),(5.0,4.0,3.0,2.0,1.0)])################
#print x
#c = 1.0  without LDA
#const = 1e-8
# for full size
'''c = 2.0 #full size linear
const = 1e-10  #full size linear '''
c = 2.0 #full size linear
const = 1e-10
const_P = 1e-13
#--------------------------------
accuracy_rates = []
for type_ in kernel_types:
    all = get_w_and_b(x,training_class,c,const, type_)
    tmp = testing_phase(x,T,testing_class, all, const, type_)
    accuracy_rates.append(tmp)
print "\n\n---------------------------------"
print "all Accuracy rates ", accuracy_rates
#----------------testing ----------------------------------------





