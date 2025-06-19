import numpy as np
import cca.prob_cca as pcca
import fa.factor_analysis as fa_mdl
import scipy.linalg as slin
import sklearn.model_selection as ms
from joblib import Parallel,delayed
from functools import partial
from psutil import cpu_count
from tqdm import tqdm

class pcca_fa:
    '''
    pCCA-FA is a dimensionality reduction framework that combines probabilistic canonical correlation analysis (pCCA) 
    and factor analysis (FA) to model across- and within- dataset interactions.

    This class implements the pCCA-FA model, stores parameters, and contains methods for fitting the model to data and computing model metrics.

    Methods
    -------
    train()
        Fit a pCCA-FA model to data using expectation-maximization (EM) algorithm.
    get_loading_matrices()
        Get across- and within-area loading matrices of the fit model.
    get_canonical_directions()
        Get canonical directions from the parameters of the fit model, as in canonical correlation analysis (CCA).
    get_correlative_modes()
        Transforms across-area loading matrices to their correlative modes.
    get_params()
        Get parameters of the fit model. 
    set_params()
        Set parameters of the model.
    estep()
        Compute expectation of the posterior, according to the E-step of the EM algorithm.
    orthogonalize()
        Orthogonalize across- and within-area loading matrices using singular value decomposition. 
    orthogonalize_latents()
        Orthogonalize latent variables (posterior means) using singular value decomposition.
    crossvalidate()
        Perform k-fold cross-validation to select hyperparameters (optimal across- and within-area dimensionality), 
            then fit a pCCA-FA model with the selected hyperparameters.
    
    Model metric methods
    -------
    compute_load_sim()
        Compute loading similarity in each across- and within-area loading matrix.
    compute_dshared()
        Compute shared dimensionality (d_shared) in each across- and within-area loading matrix.
    compute_part_ratio()
        Compute part ratio in each across- and within-area loading matrix.
    compute_psv()
        Compute percentage of shared variance (%sv) in each across- and within-area loading matrix.
    compute_metrics()
        Wrapper to compute loading similarity, d_shared, part ratio, %sv, and canonical correlations.
    '''

    def __init__(self,min_var=0.01):
        '''
        Initialize pCCA-FA model class.

                Parameters:
                        min_var (float): Used to set the variance floor, to prevent numerical underflow.
        '''
        self.params = []
        self.min_var = min_var

    def train(self,X_1,X_2,d,d1,d2,tol=1e-6,max_iter=int(1e6),verbose=False,rand_seed=None,warmstart=True,X_1_early_stop=None,X_2_early_stop=None,start_params=None):
        '''
        Fit a pCCA-FA model to data using expectation-maximization (EM) algorithm.

                Parameters:
                        X_1 (array): Array of size N (trials) x n1 (neurons), spike counts in area 1
                        X_2 (array): Array of size N (trials) x n2 (neurons), spike counts in area 2
                        d (int): Across-area dimensionality
                        d1 (int): Within-area dimensionality for area 1
                        d2 (int): Within-area dimensionality for area 2
                        tol (float): Tolerance for convergence of the EM algorithm
                        max_iter (int): Maximum number of iterations of the EM algorithm
                        verbose (bool): Flag to print out updates during training
                        rand_seed (int): Seed for random number generator, provide to ensure reproducibility
                        warmstart (bool): Whether to initialize starting parameters of EM algorithm using pCCA and FA
                        X_1_early_stop (array): Array of size N (trials) x n1 (neurons), test spike counts in area 1
                        X_2_early_stop (array): Array of size N (trials) x n2 (neurons), test spike counts in area 2
                        start_params (dict): Dictionary containing pCCA-FA model parameters to initialize EM algorithm

                Returns:
                        LL (array): Training data log likelihood at each iteration of EM algorithm
                        testLL (array): If using early_stop test data, contains test data log likelihood at each iteration of EM algorithm. Empty array otherwise.
        '''
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
        
        early_stop = not(X_1_early_stop is None) and not(X_2_early_stop is None)

        # set some useful parameters
        N,n1 = X_1.shape
        _,n2 = X_2.shape
        mu_x1,mu_x2 = X_1.mean(axis=0),X_2.mean(axis=0)
        X_1c,X_2c = (X_1-mu_x1), (X_2-mu_x2) 
        X_total = np.concatenate((X_1c,X_2c),axis=1)
        covX_1 = 1/N * (X_1c.T).dot(X_1c)
        covX_2 = 1/N * (X_2c.T).dot(X_2c)
        sampleCov = 1/N * (X_total.T).dot(X_total)
        var_floor = self.min_var*np.diag(sampleCov)
        Iz = np.identity(d+d1+d2)
        const = (n1+n2)*np.log(2*np.pi)

        if early_stop:
            X_1c_test,X_2c_test = X_1_early_stop - mu_x1, X_2_early_stop - mu_x2
            X_total_test = np.concatenate((X_1c_test,X_2c_test),axis=1)
            cov_test = (1/N)*(X_total_test.T.dot(X_total_test))

        # check that covariance is full rank
        if np.linalg.matrix_rank(sampleCov)==(n1+n2):
            x1_scale = np.exp(2/n1*np.sum(np.log(np.diag(slin.cholesky(covX_1)))))
            x2_scale = np.exp(2/n2*np.sum(np.log(np.diag(slin.cholesky(covX_2)))))
        else:
            raise np.linalg.LinAlgError(f'Covariance matrix is low rank ({np.linalg.matrix_rank(sampleCov):d}, should be {n1+n2:d})')

        # initialize parameters
        if warmstart:
            # get across-area loading matrices from pCCA and within-area loading matrices from FA
            tmp = pcca.prob_cca()
            tmp.train_maxLL(X_1,X_2,d)
            W_1 = tmp.get_params()['W_x']
            W_2 = tmp.get_params()['W_y']
            tmp = fa_mdl.factor_analysis()
            tmp.train(X_1,d1,rand_seed=rand_seed)
            L_1 = tmp.get_params()['L']
            tmp = fa_mdl.factor_analysis()
            tmp.train(X_2,d2,rand_seed=rand_seed)
            L_2 = tmp.get_params()['L']
            Psi = np.diag(sampleCov)
        elif not(start_params is None):
            # allow for specifying parameter initialization
            W_1 = start_params['W_1']
            W_2 = start_params['W_2']
            L_1 = start_params['L_1']
            L_2 = start_params['L_2']
            Psi = np.abs(np.append(start_params['psi_1'], start_params['psi_2']))
        else:
            if d > 0:
                W_1 = np.random.randn(n1,d) * np.sqrt(x1_scale/d)
                W_2 = np.random.randn(n2,d) * np.sqrt(x2_scale/d)
            else:
                W_1 = np.random.randn(n1,d)
                W_2 = np.random.randn(n2,d)
            if d1 > 0:
                L_1 = np.random.randn(n1,d1) * np.sqrt(x1_scale/d1)
            else:
                L_1 = np.random.randn(n1,d1)
            if d2 > 0:
                L_2 = np.random.randn(n2,d2) * np.sqrt(x2_scale/d2)
            else:
                L_2 = np.random.randn(n2,d2)
            Psi = np.diag(sampleCov)
        
        # define L_total - joint loading matrix
        L_top = np.concatenate((W_1,L_1,np.zeros((n1,d2))),axis=1)
        L_bottom = np.concatenate((W_2,np.zeros((n2,d1)),L_2),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        
        L_mask = np.ones(L_total.shape)
        L_mask[:n1,(d+d1):] = np.zeros((n1,d2))
        L_mask[n1:,d:(d+d1)] = np.zeros((n2,d1))

        # EM algorithm
        LL = []
        testLL = []
        for i in range(max_iter):
            # E-step: set q(z) = p(z,zx,zy|x,y)
            iPsi = np.diag(1/Psi)
            iPsiL = iPsi.dot(L_total)
            if d==0 and d1==0 and d2==0:
                iSig = iPsi
            else:
                iSig = iPsi - iPsiL.dot(slin.inv(Iz+(L_total.T).dot(iPsiL))).dot(iPsiL.T)
            iSigL = iSig.dot(L_total)
            cov_iSigL = sampleCov.dot(iSigL)
            E_zz = Iz - (L_total.T).dot(iSigL) + (iSigL.T).dot(cov_iSigL)
            
            # compute log likelihood
            logDet = 2*np.sum(np.log(np.diag(slin.cholesky(iSig))))
            curr_LL = -N/2 * (const - logDet + np.trace(iSig.dot(sampleCov)))
            LL.append(curr_LL)
            if early_stop:
                curr_testLL = -N/2 * (const - logDet + np.trace(iSig.dot(cov_test)))
                testLL.append(curr_testLL)
            if verbose:
                print('EM iteration ',i,', LL={:.2f}'.format(curr_LL))

            # check for convergence (training LL increases by less than tol, or testLL decreases)
            if i>1:
                if (LL[-1]-LL[-2])<tol or (early_stop and testLL[-1]<testLL[-2]):
                    break

            # M-step: compute new L and Psi
            if not(d==0 and d1==0 and d2==0):
                L_total = cov_iSigL.dot(slin.inv(E_zz))
            L_total = L_total * L_mask 
            Psi = np.diag(sampleCov) - np.diag(cov_iSigL.dot(L_total.T))
            Psi = np.maximum(Psi,var_floor)

        # get final parameters after convergence or max_iter
        W_1, W_2 = L_total[:n1,:d], L_total[n1:,:d]
        L_1, L_2 = L_total[:n1,d:(d+d1)], L_total[n1:,(d+d1):]
        psi_1, psi_2 = Psi[:n1], Psi[n1:]

        # create parameter dict
        self.params = {
            'mu_x1':mu_x1,'mu_x2':mu_x2, # estimated mean per neuron
            'L_total':L_total, # maximum likelihood estimated matrix
            'W_1':W_1,'W_2':W_2, # across-area loading matrices
            'L_1':L_1,'L_2':L_2, # within-area loading matrices
            'psi_1':psi_1,'psi_2':psi_2, # private variance per neuron
            'd':d,'d1':d1,'d2':d2, # selected dimensionalities
        }
        
        return np.array(LL), np.array(testLL)
    
    def get_loading_matrices(self):
        '''
        Get across- and within-area loading matrices of the fit model.

                Returns:
                        W_1 (array): Array of size n1 (neurons) x d (latents) containing the loadings for across-area latent variables onto neurons in area 1
                        W_2 (array): Array of size n2 (neurons) x d (latents) containing the loadings for across-area latent variables onto neurons in area 2
                        L_1 (array): Array of size n1 (neurons) x d1 (latents) containing the loadings for within-area latent variables onto neurons in area 1
                        L_2 (array): Array of size n2 (neurons) x d2 (latents) containing the loadings for within-area latent variables onto neurons in area 2
        '''

        n1 = len(self.params['mu_x1'])
        d, d1 = self.params['d'], self.params['d1']
        L_total = self.params['L_total']

        # get final parameters
        W_1, W_2 = L_total[:n1,:d], L_total[n1:,:d]
        L_1, L_2 = L_total[:n1,d:(d+d1)], L_total[n1:,(d+d1):]

        return W_1, W_2, L_1, L_2
    
    def get_canonical_directions(self):
        '''
        Get canonical directions from the parameters of the fit model, as in canonical correlation analysis (CCA).

                Returns:
                        canonical_dirs_x (array): Array of size n1 (neurons) x d (latents) whose columns contain the canonical directions for area 1
                        canonical_dirs_y (array): Array of size n2 (neurons) x d (latents) whose columns contain the canonical directions for area 2
                        rho (array): Array of size d (latents) x 1 containing the corresponding canonical correlations
        '''

        W_1, W_2, L_1, L_2 = self.get_loading_matrices()
        psi_1, psi_2 = self.params['psi_1'], self.params['psi_2']
        d = self.params['d']

        # compute canonical correlations
        est_covX_1 = W_1.dot(W_1.T) + L_1.dot(L_1.T) + np.diag(psi_1)
        est_covX_2 = W_2.dot(W_2.T) + L_2.dot(L_2.T) + np.diag(psi_2)
        est_covX_1X_2 = W_1.dot(W_2.T)
        inv_sqrt_covX_1 = slin.inv(slin.sqrtm(est_covX_1))
        inv_sqrt_covX_2 = slin.inv(slin.sqrtm(est_covX_2))
        K = inv_sqrt_covX_1.dot(est_covX_1X_2).dot(inv_sqrt_covX_2)
        u,s,vt = slin.svd(K)
        rho = s[0:d]

        canonical_dirs_x = slin.inv(slin.sqrtm(est_covX_1)) @ u[:,:d]
        canonical_dirs_y = slin.inv(slin.sqrtm(est_covX_2)) @ vt[:d,:].T

        return (canonical_dirs_x, canonical_dirs_y), rho
    
    def get_correlative_modes(self):
        '''
        Transforms across-area loading matrices to their correlative modes.

        Follows equations in Bach & Jordan, 2005.

                Returns:
                        CorrModes_x (array): Array of size n1 (neurons) x d (latents) whose columns contain the correlative modes for area 1
                        CorrModes_y (array): Array of size n2 (neurons) x d (latents) whose columns contain the correlative modes for area 2
        '''

        W_1, W_2, L_1, L_2 = self.get_loading_matrices()
        psi_1, psi_2 = self.params['psi_1'], self.params['psi_2']
        d = self.params['d']

        # compute canonical correlations
        est_covX_1 = W_1.dot(W_1.T) + L_1.dot(L_1.T) + np.diag(psi_1)
        est_covX_2 = W_2.dot(W_2.T) + L_2.dot(L_2.T) + np.diag(psi_2)
        est_covX_1X_2 = W_1.dot(W_2.T)
        inv_sqrt_covX_1 = slin.inv(slin.sqrtm(est_covX_1))
        inv_sqrt_covX_2 = slin.inv(slin.sqrtm(est_covX_2))
        K = inv_sqrt_covX_1.dot(est_covX_1X_2).dot(inv_sqrt_covX_2)
        u,s,vt = slin.svd(K)
        rho = s[0:d]
        
        # order W_1, W_2 by canon corrs
        pd = np.diag(np.sqrt(rho))
        CorrModes_x = slin.sqrtm(est_covX_1).dot(u[:,0:d]).dot(pd)
        CorrModes_y = slin.sqrtm(est_covX_2).dot(vt[0:d,:].T).dot(pd)

        return CorrModes_x, CorrModes_y

    def get_params(self):
        '''
        Get parameters of the fit model.

                Returns:
                        params (dict): Dictionary containing each parameter of the pCCA-FA model
        '''
        return self.params

    def set_params(self,params):
        '''
        Set parameters of the model.

                Parameters:
                        params (dict): Dictionary containing each parameter of the pCCA-FA model
        '''
        self.params = params

    def estep(self,X_1,X_2):
        '''
        Compute expectation of the posterior, according to the E-step of the EM algorithm.

                Parameters:
                        X_1 (array): Array of size N (trials) x n1 (neurons), spike counts in area 1
                        X_2 (array): Array of size N (trials) x n2 (neurons), spike counts in area 2

                Returns:
                        z (dict): Dictionary containing the mean and covariance of the posterior
                        LL (float): Log likelihood of the provided spike counts X_1 and X_2 under the fit model parameters
        '''

        N,n1 = X_1.shape
        _,n2 = X_2.shape

        d,d1,d2 = self.params['d'],self.params['d1'],self.params['d2']

        # get model parameters
        mu_x1,mu_x2 = self.params['mu_x1'],self.params['mu_x2']
        L_total = self.params['L_total']
        psi_1 = self.params['psi_1']
        psi_2 = self.params['psi_2']
        Psi = np.diag(np.concatenate((psi_1,psi_2)))

        # center data and compute covariances
        X_1c = X_1-mu_x1
        X_2c = X_2-mu_x2
        X_total = np.concatenate((X_1c,X_2c),axis=1)
        sampleCov = 1/N * (X_total.T).dot(X_total)

        # compute z
        Iz = np.identity(d+d1+d2)
        C = L_total.dot(L_total.T) + Psi
        invC = slin.inv(C)
        z_mu = X_total.dot(invC).dot(L_total)
        z_cov = np.diag(np.diag(Iz - (L_total.T).dot(invC).dot(L_total)))

        # compute LL
        const = (n1+n2)*np.log(2*np.pi)
        logDet = 2*np.sum(np.log(np.diag(slin.cholesky(C))))
        LL = -N/2 * (const + logDet + np.trace(invC.dot(sampleCov)))
        
        # return posterior and LL
        z = { 
            'z_mu':z_mu[:,:d],
            'z_cov':z_cov[:d,:d],
            'zx1_mu':z_mu[:,d:(d+d1)],
            'zx1_cov':z_cov[d:(d+d1),d:(d+d1)],
            'zx2_mu':z_mu[:,(d+d1):],
            'zx2_cov':z_cov[(d+d1):,(d+d1):],
        }
        return z, LL

    def orthogonalize(self,across_mode='paired'):
        '''
        Orthogonalize across- and within-area loading matrices using singular value decomposition. 

        Note: this also transforms loading matrices to be in covariant modes (as opposed to correlative modes)

                Parameters:
                        across_mode (str): Parameter to indicate whether to orthogonalize the across-area loading matrices jointly ('paired') or individually in each area ('unpaired')
                
                Returns:
                        W_1_norm (array): Array of size n1 (neurons) x d (latents) containing orthogonal columns with the loadings for across-area latent variables onto neurons in area 1
                        W_2_norm (array): Array of size n2 (neurons) x d (latents) containing orthogonal columns the loadings for across-area latent variables onto neurons in area 2
                        L_1_norm (array): Array of size n1 (neurons) x d1 (latents) containing orthogonal columns the loadings for within-area latent variables onto neurons in area 1
                        L_2_norm (array): Array of size n2 (neurons) x d2 (latents) containing orthogonal columns the loadings for within-area latent variables onto neurons in area 2
        '''

        n1 = len(self.params['mu_x1'])
        W_1, W_2, L_1, L_2 = self.get_loading_matrices() # output from maximum likelihood estimation

        # within-area loading matrices
        u,s,_ = slin.svd(L_1,full_matrices=False)
        L_1_norm = u @ np.diag(s)
        u,s,_ = slin.svd(L_2,full_matrices=False)
        L_2_norm = u @ np.diag(s)

        # across-area loading matrices
        if across_mode == 'paired':
            W_total = np.concatenate((W_1,W_2),axis=0)
            u,s,_ = slin.svd(W_total,full_matrices=False)
            W_1_norm = u[:n1,:] @ np.diag(s)
            W_2_norm = u[n1:,:] @ np.diag(s)
        elif across_mode == 'unpaired':
            u,s,_ = slin.svd(W_1,full_matrices=False)
            W_1_norm = u @ np.diag(s)
            u,s,_ = slin.svd(W_2,full_matrices=False)
            W_2_norm = u @ np.diag(s)
        else:
            raise ValueError('across-mode must be "paired" or "unpaired"')
        
        return W_1_norm, W_2_norm, L_1_norm, L_2_norm
    
    def orthogonalize_latents(self,zx_mu,zy_mu,do_across=False,z_mu=None,across_mode='paired'):
        '''
        Orthogonalize latent variables (posterior means) using singular value decomposition.

                Parameters:
                        zx_mu (array): Array of size N (trials) x d1 (latents) containing the within-area latent variables or posterior mean in area 1
                        zy_mu (array): Array of size N (trials) x d2 (latents) containing the within-area latent variables or posterior mean in area 2
                        do_across (bool): Whether to orthogonalize the across-area latent variables (True) or not (False)
                        z_mu (array): Array of size N (trials) x d (latents) containing the across-area latent variables or posterior mean. Only used if do_across is True
                        across_mode (str): Parameter to indicate whether to orthogonalize the across-area latent variables jointly ('paired') or individually in each area ('unpaired'). Only used if do_across is True
                
                Returns:
                        z_orth (dict): Dictionary containing the orthogonalized latent variables
                        W_orth (dict): Dictionary containing the orthogonalized loading matrices
        '''

        W_1, W_2, L_1, L_2 = self.get_loading_matrices() # output from maximum likelihood estimation
        n1 = L_1.shape[0]

        # orthogonalize zx
        u,s,vt = slin.svd(L_1,full_matrices=False)
        L_1_orth = u
        TT = np.diag(s).dot(vt)
        z_x1 = (TT.dot(zx_mu.T)).T

        # orthogonalize zy
        u,s,vt = slin.svd(L_2,full_matrices=False)
        L_2_orth = u
        TT = np.diag(s).dot(vt)
        z_x2 = (TT.dot(zy_mu.T)).T

        # orthogonalize across-area
        across_z_orth = {}
        if do_across:
            if across_mode == 'paired':
                # orthogonalize across-area latents using both area's loading matrix
                W_total = np.concatenate((W_1,W_2),axis=0)
                u,s,vt = slin.svd(W_total,full_matrices=False)
                W_1_orth = u[:n1,:]
                W_2_orth = u[n1:,:]
                TT = np.diag(s).dot(vt)
                z = (TT.dot(z_mu.T)).T
                across_z_orth['area1'] = z
                across_z_orth['area2'] = z
            elif across_mode == 'unpaired':
                # orthogonalize across-area latents using each area's loading matrix
                u,s,vt = slin.svd(W_1,full_matrices=False)
                W_1_orth = u
                TT = np.diag(s).dot(vt)
                z = (TT.dot(z_mu.T)).T
                across_z_orth['area1'] = z

                u,s,vt = slin.svd(W_2,full_matrices=False)
                W_2_orth = u
                TT = np.diag(s).dot(vt)
                z = (TT.dot(z_mu.T)).T
                across_z_orth['area2'] = z
            else:
                raise ValueError('across-mode must be "paired" or "unpaired"')

        # return z_orth, W_orth
        z_orth = {
            'z':across_z_orth, # across area latent variables, empty if do_across is False
            'z1':z_x1, # within-area latent variables for area 1
            'z2':z_x2  # within-area latent variables for area 2
            }
        W_orth = {
            'W_1':W_1_orth, # across-area loading matrix for area 1
            'W_2':W_2_orth, # across-area loading matrix for area 2
            'L_1':L_1_orth, # within-area loading matrix for area 1
            'L_2':L_2_orth # within-area loading matrix for area 2
            }

        return z_orth, W_orth

    def crossvalidate(self,X_1,X_2,d_list=np.linspace(0,8,9),d1_list=np.linspace(0,8,9),d2_list=np.linspace(0,8,9),n_folds=10,verbose=True,max_iter=int(1e6),tol=1e-6,warmstart=True,rand_seed=None,parallelize=False,early_stop=False):
        '''
        Perform k-fold cross-validation to select hyperparameters (optimal across- and within-area dimensionality), then fit a pCCA-FA model with the selected hyperparameters.

                Parameters:
                        X_1 (array): Array of size N (trials) x n1 (neurons), spike counts in area 1
                        X_2 (array): Array of size N (trials) x n2 (neurons), spike counts in area 2
                        d_list (array): 1-dimensional array containing the across-area dimensionalities to test
                        d1_list (array): 1-dimensional array containing the within-area dimensionalities to test for area 1
                        d2_list (array): 1-dimensional array containing the within-area dimensionalities to test for area 2
                        n_folds (int): The number of folds (k) for cross-validation
                        verbose (bool): Flag to print out updates during training
                        max_iter (int): Maximum number of iterations of the EM algorithm
                        tol (float): Tolerance for convergence of the EM algorithm
                        warmstart (bool): Whether to initialize starting parameters of EM algorithm using pCCA and FA
                        rand_seed (int): Seed for random number generator, provide to ensure reproducibility
                        parallelize (bool): Whether to parallelize cross-validation folds (True) or not (False), to reduce run time
                        early_stop (bool): Whether to use early_stop (True) or not (False) on the testing data of each cross-validation fold

                Returns:
                        results (dict): Dictionary containing the lists of tested dimensionalities and their corresponding cross-validated data log likelihood and prediction errors, 
                                          as well as the selected dimensionalities and its corresponding cross-validated log likelihood
        '''

        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        N = X_1.shape[0]

        # make sure z dims are integers
        d_list,d1_list,d2_list = np.meshgrid(d_list.astype(int),d1_list.astype(int),d2_list.astype(int))
        d_list = np.matrix.flatten(d_list)
        d1_list = np.matrix.flatten(d1_list)
        d2_list = np.matrix.flatten(d2_list)
        results = {'d_list':d_list,'d1_list':d1_list,'d2_list':d2_list}

        # create k-fold iterator
        if verbose:
            print('Crossvalidating pCCA-FA model to choose # of dims...')
        cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

        # iterate through train/test splits
        i = 0
        LLs,PEs = np.zeros([n_folds,len(d_list)]),np.zeros([n_folds,len(d_list)])
        for train_idx,test_idx in cv_kfold.split(X_1):
            if verbose:
                print('   Fold ',i+1,' of ',n_folds,'...')
            X_1_train,X_1_test = X_1[train_idx], X_1[test_idx]
            X_2_train,X_2_test = X_2[train_idx], X_2[test_idx]
            
            # iterate through each d, provide training and testing trials to the helper function
            func = partial(self._cv_helper,X_1train=X_1_train,X_2train=X_2_train,X_1test=X_1_test,X_2test=X_2_test,\
                           rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart,early_stop=early_stop)
            if parallelize:
                tmp = Parallel(n_jobs=cpu_count(logical=False),backend='loky')\
                    (delayed(func)(d_list[j],d1_list[j],d2_list[j]) for j in range(len(d_list)))
                LLs[i,:] = [val[0] for val in tmp]
                PEs[i,:] = [val[1] for val in tmp]
            else:
                for j in tqdm(range(len(d_list))):
                    tmp = func(d_list[j],d1_list[j],d2_list[j])
                    LLs[i,j],PEs[i,j] = tmp[0],tmp[1]
                    
            i = i+1
        
        sum_LLs = LLs.sum(axis=0)
        sum_SEs = PEs.sum(axis=0)
        results['LLs'] = sum_LLs
        results['PEs'] = sum_SEs

        # find the best # of z dimensions and train final pCCA-FA model
        max_idx = np.argmax(sum_LLs)
        d,d1,d2 = d_list[max_idx],d1_list[max_idx],d2_list[max_idx]
        results['d']=d
        results['d1']=d1
        results['d2']=d2
        results['final_LL'] = sum_LLs[max_idx]
        self.train(X_1,X_2,d,d1,d2)

        # cross-validate to get cross-validated canonical correlations
        if verbose:
            print('Crossvalidating pCCA-FA model to compute canon corrs...')
        zx1,zx2 = np.zeros((2,N,d))
        for train_idx,test_idx in cv_kfold.split(X_1):
            X_1_train,X_1_test = X_1[train_idx], X_1[test_idx]
            X_2_train,X_2_test = X_2[train_idx], X_2[test_idx]

            tmp = pcca_fa()
            tmp.train(X_1_train,X_2_train,d,d1,d2,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart)
            W_1,W_2,L_1,L_2 = tmp.get_loading_matrices() # take direct EM outputs to compute E-step
            tmp_params = tmp.get_params()
            
            # compute pCCA E-step: E[z|x] and E[z|y]
            X_1c = X_1_test - tmp_params['mu_x1']
            Cx1 = W_1 @ W_1.T + (L_1 @ L_1.T + np.diag(tmp_params['psi_1']))
            invCx1 = slin.inv(Cx1)
            zx1_mu = X_1c.dot(invCx1).dot(W_1)

            X_2c = X_2_test - tmp_params['mu_x2']
            Cx2 = W_2 @ W_2.T + (L_2 @ L_2.T + np.diag(tmp_params['psi_2']))
            invCx2 = slin.inv(Cx2)
            zx2_mu = X_2c.dot(invCx2).dot(W_2)

            zx1[test_idx,:] = zx1_mu
            zx2[test_idx,:] = zx2_mu

        cv_rho = np.zeros(d)
        for i in range(d):
            tmp = np.corrcoef(zx1[:,i],zx2[:,i])
            cv_rho[i] = tmp[0,1]
        
        self.params['cv_rho'] = cv_rho
        self.cv_results = results

        return results

    def _cv_helper(self,d,d1,d2,X_1train,X_2train,X_1test,X_2test,rand_seed=None,max_iter=int(1e5),tol=1e-6,warmstart=True,early_stop=False):
        '''
        Helper function for crossvalidate().

        Runs one train-test split and computes the log-likelihood and prediction error on the testing data.

                Parameters:
                        d (int): Across-area dimensionality
                        d1 (int): Within-area dimensionality for area 1
                        d2 (int): Across-area dimensionality for area 2
                        X_1train (array): Array of size Ntrain (trials) x n1 (neurons), training spike counts in area 1
                        X_2train (array): Array of size Ntrain (trials) x n2 (neurons), training spike counts in area 2
                        X_1test (array): Array of size Ntest (trials) x n1 (neurons), testing spike counts in area 1
                        X_2test (array): Array of size Ntest (trials) x n2 (neurons), testing spike counts in area 2
                        rand_seed (int): Seed for random number generator, provide to ensure reproducibility
                        max_iter (int): Maximum number of iterations of the EM algorithm
                        tol (float): Tolerance for convergence of the EM algorithm
                        warmstart (bool): Whether to initialize starting parameters of EM algorithm using pCCA and FA
                        early_stop (bool): Whether to use early_stop (True) or not (False) on the testing data

                Returns:
                        LL (float): Cross-validated data log likelihood of the testing data
                        PE (float): Prediction error of the testing data using leave-one-out prediction
        '''
        
        tmp = pcca_fa()
        if early_stop:
            tmp.train(X_1train,X_2train,d,d1,d2,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart,X_1_early_stop=X_1test,X_2_early_stop=X_2test)
        else:
            tmp.train(X_1train,X_2train,d,d1,d2,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart)
        # log-likelihood
        _,LL = tmp.estep(X_1test,X_2test)
        # prediction error
        X_1test_pred,X_2test_pred = tmp._leaveoneout_pred(X_1test,X_2test)
        PE = np.sum(np.square(X_1test_pred - X_1test)) + np.sum(np.square(X_2test_pred - X_2test))
        
        return (LL,PE)

    def _leaveoneout_pred(self,X_1,X_2):
        '''
        Helper function for crossvalidate().

        Runs leave-one-out prediction on provided data.

                Parameters:
                        X_1 (array): Array of size N (trials) x n1 (neurons), spike counts in area 1
                        X_2 (array): Array of size N (trials) x n2 (neurons), spike counts in area 2

                Returns:
                        pred_x (array): Array of size N (trials) x n1 (neurons) containing the prediction errors for area 1
                        pred_y (array): Array of size N (trials) x n2 (neurons) containing the prediction errors for area 2
        '''
        
        N,n1 = X_1.shape # trials x neurons
        n2 = X_2.shape[1]
        X_total = np.concatenate((X_1,X_2),axis=1)

        # extract model parameters
        Psi = np.concatenate((self.params['psi_1'],self.params['psi_2']),axis=0)
        mu = np.concatenate((self.params['mu_x1'],self.params['mu_x2']),axis=0)
        L_total = self.params['L_total']

        # compute covariances
        mdl_cov = L_total.dot(L_total.T) + np.diag(Psi)
        inv_cov = slin.inv(mdl_cov)

        # compute conditional expectations (predictions)
        n_total = n1+n2
        pred_total = np.zeros((N,n_total))
        for i in range(n_total):
            E = np.delete(np.delete(inv_cov,i,axis=0),i,axis=1)
            f = np.delete(inv_cov[:,i],i,axis=0)
            h = inv_cov[i,i]
            inv_term = E - (1/h)*np.outer(f,f)
            proj_term = np.delete(mdl_cov[i,:],i) # 1 x n_total-1
            mean_term = np.delete(X_total,i,axis=1) - np.delete(mu,i,axis=0).T # N x n_total-1
            pred = mu[i] + proj_term.dot(inv_term.dot(mean_term.T)) # 1 x N
            pred_total[:,i] = pred.T

        pred_x = pred_total[:,:n1] # predictions for neurons in area 1
        pred_y = pred_total[:,n1:] # predictions for neurons in area 2

        return pred_x,pred_y

    def compute_load_sim(self):
        '''
        Compute loading similarity in each across- and within-area loading matrix.

                Returns:
                        ls (dict): Dictionary containing the loading similarity for each across- and within-area loading matrix

        '''
        n1 = self.params['W_1'].shape[0]
        n2 = self.params['W_2'].shape[0]

        # first, orthonormalize each loading matrix
        W_1,_,_ = slin.svd(self.params['W_1'],full_matrices=False)
        W_2,_,_ = slin.svd(self.params['W_2'],full_matrices=False)
        L_1,_,_ = slin.svd(self.params['L_1'],full_matrices=False)
        L_2,_,_ = slin.svd(self.params['L_2'],full_matrices=False)

        # calculate loading similarity - following equation in Umakantha, Morina, Cowley, et al., 2021.
        ls_W_1 = 1 - n1*W_1.var(axis=0,ddof=0)
        ls_W_2 = 1 - n2*W_2.var(axis=0,ddof=0)
        ls_L_1 = 1 - n1*L_1.var(axis=0,ddof=0)
        ls_L_2 = 1 - n2*L_2.var(axis=0,ddof=0)

        ls = {
            'ls_W_1':ls_W_1, # across-area loading similarity for area 1, involves W_1
            'ls_W_2':ls_W_2, # across-area loading similarity for area 2, involves W_2
            'ls_L_1':ls_L_1, # within-area loading similarity for area 1, involves L_1
            'ls_L_2':ls_L_2  # within-area loading similarity for area 2, involves L_2
        }
        return ls

    def compute_dshared(self,cutoff_thresh=0.95):
        '''
        Compute shared dimensionality (d_shared) in each across- and within-area loading matrix.

                Parameters:
                        cutoff_thresh (float): Cutoff percentage (0-1) of across- or within-area shared variance to explain for selecting d_shared

                Returns:
                        dshared (dict): Dictionary containing the across- and within-area d_shared for each area
        '''

        W_1,W_2,L_1,L_2 = self.get_loading_matrices()

        # for across-area
        if self.params['d'] > 0:
            # area 1
            shared = W_1.dot(W_1.T)
            s = slin.svdvals(shared) # eigenvalues of WWT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_W_1 = dims[0]+1

            # area 2
            shared = W_2.dot(W_2.T)
            s = slin.svdvals(shared) # eigenvalues of WWT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_W_2 = dims[0]+1

            # overall
            W_total = np.concatenate((W_1,W_2),axis=0)
            shared = W_total.dot(W_total.T)
            s = slin.svdvals(shared) # eigenvalues of WWT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_W_total = dims[0]+1
        else:
            dshared_W_1 = 0
            dshared_W_2 = 0
            dshared_W_total = 0

        # for within area 1
        if self.params['d1'] > 0:
            shared = L_1.dot(L_1.T)
            s = slin.svdvals(shared) # eigenvalues of LLT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_L_1 = dims[0]+1
        else:
            dshared_L_1 = 0

        # for within area 2
        if self.params['d2'] > 0:
            shared = L_2.dot(L_2.T)
            s = slin.svdvals(shared) # eigenvalues of LLT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_L_2 = dims[0]+1
        else:
            dshared_L_2 = 0

        dshared = {
            'dshared_W_1':dshared_W_1, # d_shared for across-area shared variance in area 1, involves W_1
            'dshared_W_2':dshared_W_2, # d_shared for across-area shared variance in area 2, involves W_2
            'dshared_L_1':dshared_L_1, # d_shared for within-area shared variance in area 1, involves L_1
            'dshared_L_2':dshared_L_2, # d_shared for within-area shared variance in area 2, involves L_2
            'dshared_W_total':dshared_W_total # d_shared for across-area shared variance jointly for area 1 and 2, involves W_1 and W_2
        }

        return dshared

    def compute_part_ratio(self):
        '''
        Compute part ratio in each across- and within-area loading matrix.

                Returns:
                        pr (dict): Dictionary containing the part ratio for each across- and within-area loading matrix
        '''

        W_1,W_2,L_1,L_2 = self.get_loading_matrices()

        # for across-area
        shared = W_1.dot(W_1.T)
        s = slin.svdvals(shared)
        pr_W_1 = np.square(s.sum()) / np.square(s).sum()

        shared = W_2.dot(W_2.T)
        s = slin.svdvals(shared)
        pr_W_2 = np.square(s.sum()) / np.square(s).sum()

        # overall
        W_total = np.concatenate((W_1,W_2),axis=0)
        shared = W_total.dot(W_total.T)
        s = slin.svdvals(shared)
        pr_W_total = np.square(s.sum()) / np.square(s).sum()

        # for within area 1
        shared = L_1.dot(L_1.T)
        s = slin.svdvals(shared)
        pr_L_1 = np.square(s.sum()) / np.square(s).sum()

        # for within area 2
        shared = L_2.dot(L_2.T)
        s = slin.svdvals(shared)
        pr_L_2 = np.square(s.sum()) / np.square(s).sum()

        pr = {
            'pr_W_1':pr_W_1, # part ratio for across-area loading matrix in area 1, involves W_1
            'pr_W_2':pr_W_2, # part ratio for across-area loading matrix in area 2, involves W_2
            'pr_L_1':pr_L_1, # part ratio for within-area loading matrix in area 1, involves L_1
            'pr_L_2':pr_L_2, # part ratio for within-area loading matrix in area 2, involves L_2
            'pr_W_total':pr_W_total # part ratio for across-area loading matrix jointly for area 1 and 2, involves W_1 and W_2
        }

        return pr

    def compute_psv(self):
        '''
        Compute percentage of shared variance (%sv) in each across- and within-area loading matrix.

                Returns:
                        psv (dict): Dictionary containing the across- and within-area %sv for neurons in each area 
        '''

        W_1,W_2,L_1,L_2 = self.get_loading_matrices()
        psi_1,psi_2 = self.params['psi_1'],self.params['psi_2']
        
        shared_across_x1 = np.diag(W_1.dot(W_1.T))
        shared_across_x2 = np.diag(W_2.dot(W_2.T))
        shared_within_x1 = np.diag(L_1.dot(L_1.T))
        shared_within_x2 = np.diag(L_2.dot(L_2.T))
        total_x = shared_across_x1 + shared_within_x1 + psi_1
        total_y = shared_across_x2 + shared_within_x2 + psi_2
        
        # for area 1
        psv_W_1 = (shared_across_x1 / total_x).flatten() * 100
        psv_L_1 = (shared_within_x1 / total_x).flatten() * 100
        ind_var_x1 = (psi_1 / total_x).flatten() * 100
        avg_psv_W_1 = np.mean(psv_W_1)
        avg_psv_L_1 = np.mean(psv_L_1)

        # for area 2
        psv_W_2 = (shared_across_x2 / total_y).flatten() * 100
        psv_L_2 = (shared_within_x2 / total_y).flatten() * 100
        ind_var_x2 = (psi_2 / total_y).flatten() * 100
        avg_psv_W_2 = np.mean(psv_W_2)
        avg_psv_L_2 = np.mean(psv_L_2)

        # overall
        avg_psv_W_total = np.mean(np.concatenate((psv_W_1,psv_W_2)))
        avg_psv_L_total = np.mean(np.concatenate((psv_L_1,psv_L_2)))

        psv = {
            'psv_W_1':psv_W_1, # percent of across-area shared variance for each neuron in area 1
            'psv_W_2':psv_W_2, # percent of across-area shared variance for each neuron in area 2
            'psv_L_1':psv_L_1, # percent of within-area shared variance for each neuron in area 1
            'psv_L_2':psv_L_2, # percent of within-area shared variance for each neuron in area 2
            'avg_psv_W_1':avg_psv_W_1, # percent of across-area shared variance, averaged across neurons in area 1
            'avg_psv_W_2':avg_psv_W_2, # percent of across-area shared variance, averaged across neurons in area 1
            'avg_psv_L_1':avg_psv_L_1, # percent of across-area shared variance, averaged across neurons in area 1
            'avg_psv_L_2':avg_psv_L_2, # percent of across-area shared variance, averaged across neurons in area 1
            'ind_var_x1':ind_var_x1, # percent of independent variance for each neuron in area 1
            'ind_var_x2':ind_var_x2, # percent of independent variance for each neuron in area 2
            'avg_psv_W_total':avg_psv_W_total, # percent of across-area shared variance, averaged across all neurons
            'avg_psv_L_total':avg_psv_L_total, # percent of within-area shared variance, averaged across all neurons
        }
        return psv

    def compute_metrics(self,cutoff_thresh=0.95):
        '''
        Wrapper to compute loading similarity, d_shared, part ratio, %sv, and canonical correlations.

                Returns:
                        metrics (dict): Dictionary containing the computed metrics (loading similarity, d_shared, part ratio, %sv, and canonical correlations)
        '''

        dshared = self.compute_dshared(cutoff_thresh=cutoff_thresh)
        psv = self.compute_psv()
        pr = self.compute_part_ratio()
        ls = self.compute_load_sim()
        _,rho = self.get_canonical_directions()

        metrics = {
            'dshared':dshared, # dictionary of d_shared metric
            'psv':psv,         # dictionary of %sv metric
            'part_ratio':pr,   # dictionary of part ratio metric
            'load_sim':ls,     # dictionary of loading similarity metric
            'rho':rho          # array of canonical correlations
        }
        if 'cv_rho' in self.params:
            metrics['cv_rho'] = self.params['cv_rho'] # array of cross-validated canonical correlations (if crossvalidate() was called)

        return metrics