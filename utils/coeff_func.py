from cgi import print_form
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import fmin
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.optimize import leastsq

def logistic(t, x):
    return 0.5 - (1 / (1 + np.exp(t * x)))

def fitfun(t, x):
    res = t[0] * (logistic(t[1], (x-t[2]))) + t[3] + t[4] * x
    return res

def errfun(t, x, y):
    return np.sum(np.power(y - fitfun(t, x),2))

def fitfun_4para(t, x):
    res = t[0] * (logistic(t[1], (x-t[2]))) + t[3]
    return res

def errfun_4para(t, x, y):
    return np.sum(np.power(y - fitfun(t, x),2))

def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse

def coeff_fit(Obj,y) :
    temp = pearsonr(Obj, y)
    t = np.zeros(5)
    t[2] = np.mean(Obj)
    t[3] = np.mean(y)
    t[1] = 1/np.std(Obj)
    t[0] = abs(np.max(y) - np.min(y))
    t[4] = -1
    signslope = 1
    if temp[1]<=0:
        t[0] *= -1
        signslope *= -1
    v = [t, Obj, y]
    tt = fmin(errfun, t, args=(Obj, y))
    fit = fitfun(tt, Obj)
    cc = pearsonr(fit, y)[0]
    # print("plcc")
    srocc = spearmanr(fit, y).correlation
    # print("srcc")
    krocc = kendalltau(fit, y).correlation
    # print("krocc")
    rmse = RMSE( np.absolute(y), np.absolute(fit) )
    # print("Rmse")
    return fit, cc, srocc, krocc, rmse

def compute_stress(de,dv): #obj->delta E y->subjective->dV
    fcv = np.sum(de*de)/np.sum(de*dv)
    STRESS = 100*sqrt(np.sum((de-fcv*dv)*(de-fcv*dv))/(fcv*fcv*np.sum(dv*dv)))
    return STRESS

if __name__ == '__main__':
    data=pd.read_csv('bfd_p3_model2_cube.csv',header=0)
    a=data['score'].tolist()
    data=pd.read_csv('bfd_p3_model1_cube.csv',header=0)
    b=data['score'].tolist()
    print(compute_stress(np.array(a),np.array(b)))
    print(compute_stress(np.array(b),np.array(a)))



