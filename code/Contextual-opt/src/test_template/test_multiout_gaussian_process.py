
# URL: https://bigaidream.gitbooks.io/subsets_ml_cookbook/content/bayes/gp/coregionalized_regression_gpy.html
# URL https://nbviewer.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb

import GPy
import numpy as np
np.set_printoptions(linewidth=100)

import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.dirname(__file__) + "/../../")
from src.objective_function import benchmark as bench
from src.model.gp import gp_interface as gp
from src.model.gp import bo_acq
from src.util import sampler as sampler
from src.util import weight as weight

current_dir = os.path.dirname(__file__)

def sample_objective_function(train_num=100, test_num=100):
     #This functions generate data corresponding to two outputs
    f_output1 = lambda x: 4. * np.cos(x/5.) - .4*x - 35. + np.random.rand(x.size)[:,None] * 2.
    f_output2 = lambda x: 6. * np.cos(x/5.) + .2*x + 35. + np.random.rand(x.size)[:,None] * 8.


    #{X,Y} training set for each output
    X1 = np.random.rand(train_num)[:,None]; X1=X1*75
    X2 = np.random.rand(train_num)[:,None]; X2=X2*70 + 30
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)
    #{X,Y} test set for each output
    Xt1 = np.random.rand(test_num)[:,None]*test_num
    Xt2 = np.random.rand(test_num)[:,None]*test_num
    Yt1 = f_output1(Xt1)
    Yt2 = f_output2(Xt2)

    return X1, X2, Y1, Y2, Xt1, Xt2, Yt1, Yt2

def multiout_gp():
   
    X1, X2, Y1, Y2, Xt1, Xt2, Yt1, Yt2 = sample_objective_function(10,10)

    K1 = GPy.kern.Bias(1)
    K2 = GPy.kern.Linear(1)
    lcm = GPy.util.multioutput.LCM(input_dim=1,num_outputs=2,kernels_list=[K1,K2])

    m = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2],kernel=lcm)
    # m['.*bias.var'].constrain_fixed(1.)
    # m['.*W'].constrain_fixed(0)
    # m['.*linear.var'].constrain_fixed(1.)
    m.optimize()

    # plot result
    xlim = (0,100); ylim = (0,50)
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    ax1.plot(X1[:,:1],Y1,'kx',mew=1.5,label='Train set')
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5,label='Test set')
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    ax2.plot(X2[:,:1],Y2,'kx',mew=1.5,label='Train set')
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5,label='Test set')
    fig.savefig(os.path.join(current_dir, "gp_multioutput_test.pdf"))

    # check covariance
    newX = np.arange(100,110)#[:,None]

    # newX = np.vstack([np.c_[newX, np.zeros_like(newX)], np.c_[newX,np.ones_like(newX)]])
    newX = np.c_[newX, np.zeros_like(newX)]
    noise_dict = {'output_index':newX[:,1:].astype(int)}
    print(m.predict(newX, full_cov=False, Y_metadata=noise_dict))


def multiout_gp_interface():
       
    X1, X2, Y1, Y2, Xt1, Xt2, Yt1, Yt2 = sample_objective_function()

    model = gp.GPInterface(1, exact_feval=True, optimize_restarts=0, verbose=False)
    model._create_model(np.array([X1, X2]), np.array([Y1, Y2]))

    print(np.array([X1, X2]).shape)
    print(np.array([Y1, Y2]).shape)

    # check covariance
    newX = np.arange(100,120,5)[:,None]

    X_mean = model.predict_mean(newX)
    X_diag_sigma = model.predict_sigma(newX)
    X_cov = model.predict_covariance(newX)
    
    print("- mean --")
    print(X_mean)
    print("- std --")
    print(X_diag_sigma)
    print("- diag cov --")
    print(X_diag_sigma ** 2)
    print("- cov --")
    print(X_cov)

if __name__ == '__main__':
    multiout_gp()
    # multiout_gp_interface()