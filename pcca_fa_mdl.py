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
        Perform k-fold cross-validation to select hyperparameters (optimal across- and within-area dimensionality), then fit a pCCA-FA model with the selected hyperparameters.
    
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

    def train(self,X,Y,zDim,zxDim,zyDim,tol=1e-6,max_iter=int(1e6),verbose=False,rand_seed=None,warmstart=True,X_early_stop=None,Y_early_stop=None,start_params=None):
        '''
        Fit a pCCA-FA model to data using expectation-maximization (EM) algorithm.

                Parameters:
                        X (array): Array of size N (trials) x xDim (neurons), spike counts in area 1
                        Y (array): Array of size N (trials) x yDim (neurons), spike counts in area 2
                        zDim (int): Across-area dimensionality
                        zxDim (int): Within-area dimensionality for area 1
                        zyDim (int): Within-area dimensionality for area 2
                        tol (float): Tolerance for convergence of the EM algorithm
                        max_iter (int): Maximum number of iterations of the EM algorithm
                        verbose (bool): Flag to print out updates during training
                        rand_seed (int): Seed for random number generator, provide to ensure reproducibility
                        warmstart (bool): Whether to initialize starting parameters of EM algorithm using pCCA and FA
                        X_early_stop (array): Array of size N (trials) x xDim (neurons), test spike counts in area 1
                        Y_early_stop (array): Array of size N (trials) x yDim (neurons), test spike counts in area 2
                        start_params (dict): Dictionary containing pCCA-FA model parameters to initialize EM algorithm

                Returns:
                        LL (array): Training data log likelihood at each iteration of EM algorithm
                        testLL (array): If using early_stop test data, contains test data log likelihood at each iteration of EM algorithm. Empty array otherwise.
        '''
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
        
        early_stop = not(X_early_stop is None) and not(Y_early_stop is None)

        # set some useful parameters
        N,xDim = X.shape
        N,yDim = Y.shape
        mu_x,mu_y = X.mean(axis=0),Y.mean(axis=0)
        Xc,Yc = (X-mu_x), (Y-mu_y) 
        XcYc = np.concatenate((Xc,Yc),axis=1)
        covX = 1/N * (Xc.T).dot(Xc)
        covY = 1/N * (Yc.T).dot(Yc)
        sampleCov = 1/N * (XcYc.T).dot(XcYc)
        var_floor = self.min_var*np.diag(sampleCov)
        Iz = np.identity(zDim+zxDim+zyDim)
        const = (xDim+yDim)*np.log(2*np.pi)

        if early_stop:
            Xc_test,Yc_test = X_early_stop-mu_x,Y_early_stop-mu_y
            XcYc_test = np.concatenate((Xc_test,Yc_test),axis=1)
            cov_test = (1/N)*(XcYc_test.T.dot(XcYc_test))

        # check that covariance is full rank
        if np.linalg.matrix_rank(sampleCov)==(xDim+yDim):
            x_scale = np.exp(2/xDim*np.sum(np.log(np.diag(slin.cholesky(covX)))))
            y_scale = np.exp(2/yDim*np.sum(np.log(np.diag(slin.cholesky(covY)))))
        else:
            raise np.linalg.LinAlgError(f'Covariance matrix is low rank ({np.linalg.matrix_rank(sampleCov):d}, should be {xDim+yDim:d})')

        # initialize parameters
        if warmstart:
            tmp = pcca.prob_cca()
            tmp.train_maxLL(X,Y,zDim)
            W_x = tmp.get_params()['W_x']
            W_y = tmp.get_params()['W_y']
            tmp = fa_mdl.factor_analysis()
            tmp.train(X,zxDim,rand_seed=rand_seed)
            L_x = tmp.get_params()['L']
            tmp = fa_mdl.factor_analysis()
            tmp.train(Y,zyDim,rand_seed=rand_seed)
            L_y = tmp.get_params()['L']
            Ph = np.diag(sampleCov)
        elif not(start_params is None):
            # allow for specifying parameter initialization
            W_x = start_params['W_x']
            W_y = start_params['W_y']
            L_x = start_params['L_x']
            L_y = start_params['L_y']
            Ph = np.abs(np.append(start_params['psi_x'], start_params['psi_y']))
        else:
            if zDim > 0:
                W_x = np.random.randn(xDim,zDim) * np.sqrt(x_scale/zDim)
                W_y = np.random.randn(yDim,zDim) * np.sqrt(y_scale/zDim)
            else:
                W_x = np.random.randn(xDim,zDim)
                W_y = np.random.randn(yDim,zDim)
            if zxDim > 0:
                L_x = np.random.randn(xDim,zxDim) * np.sqrt(x_scale/zxDim)
            else:
                L_x = np.random.randn(xDim,zxDim)
            if zyDim > 0:
                L_y = np.random.randn(yDim,zyDim) * np.sqrt(y_scale/zyDim)
            else:
                L_y = np.random.randn(yDim,zyDim)
            Ph = np.diag(sampleCov)
        
        # define L_total - joint loading matrix
        L_top = np.concatenate((W_x,L_x,np.zeros((xDim,zyDim))),axis=1)
        L_bottom = np.concatenate((W_y,np.zeros((yDim,zxDim)),L_y),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        
        L_mask = np.ones(L_total.shape)
        L_mask[:xDim,(zDim+zxDim):] = np.zeros((xDim,zyDim))
        L_mask[xDim:,zDim:(zDim+zxDim)] = np.zeros((yDim,zxDim))

        # EM algorithm
        LL = []
        testLL = []
        for i in range(max_iter):
            # E-step: set q(z) = p(z,zx,zy|x,y)
            iPh = np.diag(1/Ph)
            iPhL = iPh.dot(L_total)
            if zDim==0 and zxDim==0 and zyDim==0:
                iSig = iPh
            else:
                iSig = iPh - iPhL.dot(slin.inv(Iz+(L_total.T).dot(iPhL))).dot(iPhL.T)
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

            # M-step: compute new L and Ph
            if not(zDim==0 and zxDim==0 and zyDim==0):
                L_total = cov_iSigL.dot(slin.inv(E_zz))
            L_total = L_total * L_mask 
            Ph = np.diag(sampleCov) - np.diag(cov_iSigL.dot(L_total.T))
            Ph = np.maximum(Ph,var_floor)

        # get final parameters after convergence or max_iter
        W_x, W_y = L_total[:xDim,:zDim], L_total[xDim:,:zDim]
        L_x, L_y = L_total[:xDim,zDim:(zDim+zxDim)], L_total[xDim:,(zDim+zxDim):]
        psi_x, psi_y = Ph[:xDim], Ph[xDim:]

        # create parameter dict
        self.params = {
            'mu_x':mu_x,'mu_y':mu_y, # estimated mean per neuron
            'L_total':L_total, # maximum likelihood estimated matrix
            'W_x':W_x,'W_y':W_y, # across-area loading matrices
            'L_x':L_x,'L_y':L_y, # within-area loading matrices
            'psi_x':psi_x,'psi_y':psi_y, # private variance per neuron
            'zDim':zDim,'zxDim':zxDim,'zyDim':zyDim, # selected dimensionalities
        }
        
        return np.array(LL), np.array(testLL)
    
    def get_loading_matrices(self):
        '''
        Get across- and within-area loading matrices of the fit model.

                Returns:
                        W_x (array): Array of size xDim (neurons) x zDim (latents) containing the loadings for across-area latent variables onto neurons in area 1
                        W_y (array): Array of size yDim (neurons) x zDim (latents) containing the loadings for across-area latent variables onto neurons in area 2
                        L_x (array): Array of size xDim (neurons) x zxDim (latents) containing the loadings for within-area latent variables onto neurons in area 1
                        L_y (array): Array of size yDim (neurons) x zyDim (latents) containing the loadings for within-area latent variables onto neurons in area 2
        '''

        xDim = len(self.params['mu_x'])
        zDim, zxDim = self.params['zDim'], self.params['zxDim']
        L_total = self.params['L_total']

        # get final parameters
        W_x, W_y = L_total[:xDim,:zDim], L_total[xDim:,:zDim]
        L_x, L_y = L_total[:xDim,zDim:(zDim+zxDim)], L_total[xDim:,(zDim+zxDim):]

        return W_x, W_y, L_x, L_y
    
    def get_canonical_directions(self):
        '''
        Get canonical directions from the parameters of the fit model, as in canonical correlation analysis (CCA).

                Returns:
                        canonical_dirs_x (array): Array of size xDim (neurons) x zDim (latents) whose columns contain the canonical directions for area 1
                        canonical_dirs_y (array): Array of size yDim (neurons) x zDim (latents) whose columns contain the canonical directions for area 2
                        rho (array): Array of size zDim (latents) x 1 containing the corresponding canonical correlations
        '''

        W_x, W_y, L_x, L_y = self.get_loading_matrices()
        psi_x, psi_y = self.params['psi_x'], self.params['psi_y']
        zDim = self.params['zDim']

        # compute canonical correlations
        est_covX = W_x.dot(W_x.T) + L_x.dot(L_x.T) + np.diag(psi_x)
        est_covY = W_y.dot(W_y.T) + L_y.dot(L_y.T) + np.diag(psi_y)
        est_covXY = W_x.dot(W_y.T)
        inv_sqrt_covX = slin.inv(slin.sqrtm(est_covX))
        inv_sqrt_covY = slin.inv(slin.sqrtm(est_covY))
        K = inv_sqrt_covX.dot(est_covXY).dot(inv_sqrt_covY)
        u,d,vt = slin.svd(K)
        rho = d[0:zDim]

        canonical_dirs_x = slin.inv(slin.sqrtm(est_covX)) @ u[:,:zDim]
        canonical_dirs_y = slin.inv(slin.sqrtm(est_covY)) @ vt[:zDim,:].T

        return (canonical_dirs_x, canonical_dirs_y), rho
    
    def get_correlative_modes(self):
        '''
        Transforms across-area loading matrices to their correlative modes.

        Follows equations in Bach & Jordan, 2005.

                Returns:
                        CorrModes_x (array): Array of size xDim (neurons) x zDim (latents) whose columns contain the correlative modes for area 1
                        CorrModes_y (array): Array of size yDim (neurons) x zDim (latents) whose columns contain the correlative modes for area 2
        '''

        W_x, W_y, L_x, L_y = self.get_loading_matrices()
        psi_x, psi_y = self.params['psi_x'], self.params['psi_y']
        zDim = self.params['zDim']

        # compute canonical correlations
        est_covX = W_x.dot(W_x.T) + L_x.dot(L_x.T) + np.diag(psi_x)
        est_covY = W_y.dot(W_y.T) + L_y.dot(L_y.T) + np.diag(psi_y)
        est_covXY = W_x.dot(W_y.T)
        inv_sqrt_covX = slin.inv(slin.sqrtm(est_covX))
        inv_sqrt_covY = slin.inv(slin.sqrtm(est_covY))
        K = inv_sqrt_covX.dot(est_covXY).dot(inv_sqrt_covY)
        u,d,vt = slin.svd(K)
        rho = d[0:zDim]
        
        # order W_x, W_y by canon corrs
        pd = np.diag(np.sqrt(rho))
        CorrModes_x = slin.sqrtm(est_covX).dot(u[:,0:zDim]).dot(pd)
        CorrModes_y = slin.sqrtm(est_covY).dot(vt[0:zDim,:].T).dot(pd)

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

    def estep(self,X,Y):
        '''
        Compute expectation of the posterior, according to the E-step of the EM algorithm.

                Parameters:
                        X (array): Array of size N (trials) x xDim (neurons), spike counts in area 1
                        Y (array): Array of size N (trials) x yDim (neurons), spike counts in area 2

                Returns:
                        z (dict): Dictionary containing the mean and covariance of the posterior
                        LL (float): Log likelihood of the provided spike counts X and Y under the fit model parameters
        '''

        N,xDim = X.shape
        N,yDim = Y.shape

        zDim,zxDim,zyDim = self.params['zDim'],self.params['zxDim'],self.params['zyDim']

        # get model parameters
        mu_x,mu_y = self.params['mu_x'],self.params['mu_y']
        L_total = self.params['L_total']
        psi_x = self.params['psi_x']
        psi_y = self.params['psi_y']
        psi = np.diag(np.concatenate((psi_x,psi_y)))

        # center data and compute covariances
        Xc = X-mu_x
        Yc = Y-mu_y
        XcYc = np.concatenate((Xc,Yc),axis=1)
        sampleCov = 1/N * (XcYc.T).dot(XcYc)

        # compute z
        Iz = np.identity(zDim+zxDim+zyDim)
        C = L_total.dot(L_total.T) + psi
        invC = slin.inv(C)
        z_mu = XcYc.dot(invC).dot(L_total)
        z_cov = np.diag(np.diag(Iz - (L_total.T).dot(invC).dot(L_total)))

        # compute LL
        const = (xDim+yDim)*np.log(2*np.pi)
        logDet = 2*np.sum(np.log(np.diag(slin.cholesky(C))))
        LL = -N/2 * (const + logDet + np.trace(invC.dot(sampleCov)))
        
        # return posterior and LL
        z = { 
            'z_mu':z_mu[:,:zDim],
            'z_cov':z_cov[:zDim,:zDim],
            'zx_mu':z_mu[:,zDim:(zDim+zxDim)],
            'zx_cov':z_cov[zDim:(zDim+zxDim),zDim:(zDim+zxDim)],
            'zy_mu':z_mu[:,(zDim+zxDim):],
            'zy_cov':z_cov[(zDim+zxDim):,(zDim+zxDim):],
            'z_mu_all':z_mu,
            'z_cov_all':z_cov
        }
        return z, LL

    def orthogonalize(self,across_mode='paired'):
        '''
        Orthogonalize across- and within-area loading matrices using singular value decomposition. 

        Note: this also transforms loading matrices to be in covariant modes (as opposed to correlative modes)

                Parameters:
                        across_mode (str): Parameter to indicate whether to orthogonalize the across-area loading matrices jointly ('paired') or individually in each area ('unpaired')
                
                Returns:
                        W_x_norm (array): Array of size xDim (neurons) x zDim (latents) containing orthogonal columns with the loadings for across-area latent variables onto neurons in area 1
                        W_y_norm (array): Array of size yDim (neurons) x zDim (latents) containing orthogonal columns the loadings for across-area latent variables onto neurons in area 2
                        L_x_norm (array): Array of size xDim (neurons) x zxDim (latents) containing orthogonal columns the loadings for within-area latent variables onto neurons in area 1
                        L_y_norm (array): Array of size yDim (neurons) x zyDim (latents) containing orthogonal columns the loadings for within-area latent variables onto neurons in area 2
        '''

        xDim = len(self.params['mu_x'])
        zDim, zxDim, zyDim = self.params['zDim'], self.params['zxDim'], self.params['zyDim']
        W_x, W_y, L_x, L_y = self.get_loading_matrices() # output from maximum likelihood estimation

        # within-area loading matrices
        ulx,slx,_ = slin.svd(L_x)
        L_x_norm = ulx[:,:zxDim] @ np.diag(slx[:zxDim])
        uly,sly,_ = slin.svd(L_y)
        L_y_norm = uly[:,:zyDim] @ np.diag(sly[:zyDim])

        # across-area loading matrices
        if across_mode == 'paired':
            W_total = np.concatenate((W_x,W_y),axis=0)
            uw,sw,_ = slin.svd(W_total)
            W_x_norm = uw[:xDim,:zDim] @ np.diag(sw[:zDim])
            W_y_norm = uw[xDim:,:zDim] @ np.diag(sw[:zDim])
        elif across_mode == 'unpaired':
            uwx,swx,_ = slin.svd(W_x)
            W_x_norm = uwx[:,:zDim] @ np.diag(swx[:zDim])
            uwy,swy,_ = slin.svd(W_y)
            W_y_norm = uwy[:,:zDim] @ np.diag(swy[:zDim])
        else:
            raise ValueError('across-mode must be "paired" or "unpaired"')
        
        return W_x_norm, W_y_norm, L_x_norm, L_y_norm
    
    def orthogonalize_latents(self,zx_mu,zy_mu,do_across=False,z_mu=None,across_mode='paired'):
        '''
        Orthogonalize latent variables (posterior means) using singular value decomposition.

                Parameters:
                        zx_mu (array): Array of size N (trials) x zxDim (latents) containing the within-area latent variables or posterior mean in area 1
                        zy_mu (array): Array of size N (trials) x zyDim (latents) containing the within-area latent variables or posterior mean in area 2
                        do_across (bool): Whether to orthogonalize the across-area latent variables (True) or not (False)
                        z_mu (array): Array of size N (trials) x zDim (latents) containing the across-area latent variables or posterior mean. Only used if do_across is True
                        across_mode (str): Parameter to indicate whether to orthogonalize the across-area latent variables jointly ('paired') or individually in each area ('unpaired'). Only used if do_across is True
                
                Returns:
                        z_orth (dict): Dictionary containing the orthogonalized latent variables
                        W_orth (dict): Dictionary containing the orthogonalized loading matrices
        '''

        W_x, W_y, L_x, L_y = self.get_loading_matrices() # output from maximum likelihood estimation
        xDim = L_x.shape[0]

        # orthogonalize zx
        u,s,vt = slin.svd(L_x,full_matrices=False)
        Lx_orth = u
        TT = np.diag(s).dot(vt)
        zx = (TT.dot(zx_mu.T)).T

        # orthogonalize zy
        u,s,vt = slin.svd(L_y,full_matrices=False)
        Ly_orth = u
        TT = np.diag(s).dot(vt)
        zy = (TT.dot(zy_mu.T)).T

        # orthogonalize across-area
        across_z_orth = {}
        if do_across:
            if across_mode == 'paired':
                # orthogonalize across-area latents using both area's loading matrix
                W_total = np.concatenate((W_x,W_y),axis=0)
                u,s,vt = slin.svd(W_total,full_matrices=False)
                W_x_orth = u[:xDim,:]
                W_y_orth = u[xDim:,:]
                TT = np.diag(s).dot(vt)
                z = (TT.dot(z_mu.T)).T
                across_z_orth['x'] = z
                across_z_orth['y'] = z
            elif across_mode == 'unpaired':
                # orthogonalize across-area latents using each area's loading matrix
                u,s,vt = slin.svd(W_x,full_matrices=False)
                W_x_orth = u
                TT = np.diag(s).dot(vt)
                z = (TT.dot(z_mu.T)).T
                across_z_orth['x'] = z

                u,s,vt = slin.svd(W_y,full_matrices=False)
                W_y_orth = u
                TT = np.diag(s).dot(vt)
                z = (TT.dot(z_mu.T)).T
                across_z_orth['y'] = z
            else:
                raise ValueError('across-mode must be "paired" or "unpaired"')

        # return z_orth, W_orth
        z_orth = {
            'z':across_z_orth, # across area latent variables, empty if do_across is False
            'zx':zx, # within-area latent variables for area 1
            'zy':zy # within-area latent variables for area 2
            }
        W_orth = {
            'Wx':W_x_orth, # across-area loading matrix for area 1
            'Wy':W_y_orth, # across-area loading matrix for area 2
            'Lx':Lx_orth, # within-area loading matrix for area 1
            'Ly':Ly_orth # within-area loading matrix for area 2
            }

        return z_orth, W_orth

    def crossvalidate(self,X,Y,zDim_list=np.linspace(0,8,9),zxDim_list=np.linspace(0,8,9),zyDim_list=np.linspace(0,8,9),n_folds=10,verbose=True,max_iter=int(1e6),tol=1e-6,warmstart=True,rand_seed=None,parallelize=False,early_stop=False):
        '''
        Perform k-fold cross-validation to select hyperparameters (optimal across- and within-area dimensionality), then fit a pCCA-FA model with the selected hyperparameters.

                Parameters:
                        X (array): Array of size N (trials) x xDim (neurons), spike counts in area 1
                        Y (array): Array of size N (trials) x yDim (neurons), spike counts in area 2
                        zDim_list (array): 1-dimensional array containing the across-area dimensionalities to test
                        zxDim_list (array): 1-dimensional array containing the within-area dimensionalities to test for area 1
                        zyDim_list (array): 1-dimensional array containing the within-area dimensionalities to test for area 2
                        n_folds (int): The number of folds (k) for cross-validation
                        verbose (bool): Flag to print out updates during training
                        max_iter (int): Maximum number of iterations of the EM algorithm
                        tol (float): Tolerance for convergence of the EM algorithm
                        warmstart (bool): Whether to initialize starting parameters of EM algorithm using pCCA and FA
                        rand_seed (int): Seed for random number generator, provide to ensure reproducibility
                        parallelize (bool): Whether to parallelize cross-validation folds (True) or not (False), to reduce run time
                        early_stop (bool): Whether to use early_stop (True) or not (False) on the testing data of each cross-validation fold

                Returns:
                        LL_curves (dict): Dictionary containing the lists of tested dimensionalities and their corresponding cross-validated data log likelihood and prediction errors, 
                                          as well as the selected dimensionalities and its corresponding cross-validated log likelihood
        '''

        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        N = X.shape[0]

        # make sure z dims are integers
        z_list,zx_list,zy_list = np.meshgrid(zDim_list.astype(int),zxDim_list.astype(int),zyDim_list.astype(int))
        z_list = np.matrix.flatten(z_list)
        zx_list = np.matrix.flatten(zx_list)
        zy_list = np.matrix.flatten(zy_list)
        LL_curves = {'z_list':z_list,'zx_list':zx_list,'zy_list':zy_list}

        # create k-fold iterator
        if verbose:
            print('Crossvalidating pCCA-FA model to choose # of dims...')
        cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

        # iterate through train/test splits
        i = 0
        LLs,PEs = np.zeros([n_folds,len(z_list)]),np.zeros([n_folds,len(z_list)])
        for train_idx,test_idx in cv_kfold.split(X):
            if verbose:
                print('   Fold ',i+1,' of ',n_folds,'...')
            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]
            
            # iterate through each zDim, provide training and testing trials to the helper function
            func = partial(self._cv_helper,Xtrain=X_train,Ytrain=Y_train,Xtest=X_test,Ytest=Y_test,\
                           rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart,early_stop=early_stop)
            if parallelize:
                tmp = Parallel(n_jobs=cpu_count(logical=False),backend='loky')\
                    (delayed(func)(z_list[j],zx_list[j],zy_list[j]) for j in range(len(z_list)))
                LLs[i,:] = [val[0] for val in tmp]
                PEs[i,:] = [val[1] for val in tmp]
            else:
                for j in tqdm(range(len(z_list))):
                    tmp = func(z_list[j],zx_list[j],zy_list[j])
                    LLs[i,j],PEs[i,j] = tmp[0],tmp[1]
                    
            i = i+1
        
        sum_LLs = LLs.sum(axis=0)
        sum_SEs = PEs.sum(axis=0)
        LL_curves['LLs'] = sum_LLs
        LL_curves['PEs'] = sum_SEs

        # find the best # of z dimensions and train final pCCA-FA model
        max_idx = np.argmax(sum_LLs)
        zDim,zxDim,zyDim = z_list[max_idx],zx_list[max_idx],zy_list[max_idx]
        LL_curves['zDim']=zDim
        LL_curves['zxDim']=zxDim
        LL_curves['zyDim']=zyDim
        LL_curves['final_LL'] = sum_LLs[max_idx]
        self.train(X,Y,zDim,zxDim,zyDim)

        # cross-validate to get cross-validated canonical correlations
        if verbose:
            print('Crossvalidating pCCA-FA model to compute canon corrs...')
        zx,zy = np.zeros((2,N,zDim))
        for train_idx,test_idx in cv_kfold.split(X):
            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]

            tmp = pcca_fa()
            tmp.train(X_train,Y_train,zDim,zxDim,zyDim,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart)
            W_x,W_y,L_x,L_y = tmp.get_loading_matrices() # take direct EM outputs to compute E-step
            tmp_params = tmp.get_params()
            
            # compute pCCA E-step: E[z|x] and E[z|y]
            Xc = X_test - tmp_params['mu_x']
            Cx = W_x @ W_x.T + (L_x @ L_x.T + np.diag(tmp_params['psi_x']))
            invCx = slin.inv(Cx)
            zx_mu = Xc.dot(invCx).dot(W_x)

            Yc = Y_test - tmp_params['mu_y']
            Cy = W_y @ W_y.T + (L_y @ L_y.T + np.diag(tmp_params['psi_y']))
            invCy = slin.inv(Cy)
            zy_mu = Yc.dot(invCy).dot(W_y)

            zx[test_idx,:] = zx_mu
            zy[test_idx,:] = zy_mu

        cv_rho = np.zeros(zDim)
        for i in range(zDim):
            tmp = np.corrcoef(zx[:,i],zy[:,i])
            cv_rho[i] = tmp[0,1]
        
        self.params['cv_rho'] = cv_rho
        self.LL_curves = LL_curves

        return LL_curves

    def _cv_helper(self,zDim,zxDim,zyDim,Xtrain,Ytrain,Xtest,Ytest,rand_seed=None,max_iter=int(1e5),tol=1e-6,warmstart=True,early_stop=False):
        '''
        Helper function for crossvalidate().

        Runs one train-test split and computes the log-likelihood and prediction error on the testing data.

                Parameters:
                        zDim (int): Across-area dimensionality
                        zxDim (int): Within-area dimensionality for area 1
                        zyDim (int): Across-area dimensionality for area 2
                        Xtrain (array): Array of size Ntrain (trials) x xDim (neurons), training spike counts in area 1
                        Ytrain (array): Array of size Ntrain (trials) x yDim (neurons), training spike counts in area 2
                        Xtest (array): Array of size Ntest (trials) x xDim (neurons), testing spike counts in area 1
                        Ytest (array): Array of size Ntest (trials) x yDim (neurons), testing spike counts in area 2
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
            tmp.train(Xtrain,Ytrain,zDim,zxDim,zyDim,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart,X_early_stop=Xtest,Y_early_stop=Ytest)
        else:
            tmp.train(Xtrain,Ytrain,zDim,zxDim,zyDim,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart)
        # log-likelihood
        _,LL = tmp.estep(Xtest,Ytest)
        # prediction error
        Xtest_pred,Ytest_pred = tmp._leaveoneout_pred(Xtest,Ytest)
        PE = np.sum(np.square(Xtest_pred - Xtest)) + np.sum(np.square(Ytest_pred - Ytest))
        
        return (LL,PE)

    def _leaveoneout_pred(self,X,Y):
        '''
        Helper function for crossvalidate().

        Runs leave-one-out prediction on provided data.

                Parameters:
                        X (array): Array of size N (trials) x xDim (neurons), spike counts in area 1
                        Y (array): Array of size N (trials) x yDim (neurons), spike counts in area 2

                Returns:
                        pred_x (array): Array of size N (trials) x xDim (neurons) containing the prediction errors for area 1
                        pred_y (array): Array of size N (trials) x yDim (neurons) containing the prediction errors for area 2
        '''
        
        N,xDim = X.shape
        yDim = Y.shape[1]
        X_total = np.concatenate((X,Y),axis=1)

        # extract model parameters
        W_x,L_x,W_y,L_y = self.params['W_x'],self.params['L_x'],self.params['W_y'],self.params['L_y']
        Ph = np.concatenate((self.params['psi_x'],self.params['psi_y']),axis=0)
        mu = np.concatenate((self.params['mu_x'],self.params['mu_y']),axis=0)
        zxDim,zyDim = L_x.shape[1],L_y.shape[1]
        L_top = np.concatenate((W_x,L_x,np.zeros((xDim,zyDim))),axis=1)
        L_bottom = np.concatenate((W_y,np.zeros((yDim,zxDim)),L_y),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)

        # compute covariances
        mdl_cov = L_total.dot(L_total.T) + np.diag(Ph)
        inv_cov = slin.inv(mdl_cov)

        # compute conditional expectations (predictions)
        D = xDim+yDim
        pred_total = np.zeros((N,D))
        for i in range(D):
            E = np.delete(np.delete(inv_cov,i,axis=0),i,axis=1)
            f = np.delete(inv_cov[:,i],i,axis=0)
            h = inv_cov[i,i]
            inv_term = E - (1/h)*np.outer(f,f)
            proj_term = np.delete(mdl_cov[i,:],i) # 1 x D-1
            mean_term = np.delete(X_total,i,axis=1) - np.delete(mu,i,axis=0).T # N x D-1
            pred = mu[i] + proj_term.dot(inv_term.dot(mean_term.T)) # 1 x N
            pred_total[:,i] = pred.T

        pred_x = pred_total[:,:xDim] # predictions for neurons in area 1
        pred_y = pred_total[:,xDim:] # predictions for neurons in area 2

        return pred_x,pred_y

    def compute_load_sim(self):
        '''
        Compute loading similarity in each across- and within-area loading matrix.

                Returns:
                        ls (dict): Dictionary containing the loading similarity for each across- and within-area loading matrix

        '''
        n_x = self.params['W_x'].shape[0]
        n_y = self.params['W_y'].shape[0]

        # first, orthonormalize each loading matrix
        Wx,_,_ = slin.svd(self.params['W_x'],full_matrices=False)
        Wy,_,_ = slin.svd(self.params['W_y'],full_matrices=False)
        Lx,_,_ = slin.svd(self.params['L_x'],full_matrices=False)
        Ly,_,_ = slin.svd(self.params['L_y'],full_matrices=False)

        # calculate loading similarity - following equation in Umakantha, Morina, Cowley, et al., 2021.
        ls_x = 1 - n_x*Wx.var(axis=0,ddof=0)
        ls_y = 1 - n_y*Wy.var(axis=0,ddof=0)
        ls_priv_x = 1 - n_x*Lx.var(axis=0,ddof=0)
        ls_priv_y = 1 - n_y*Ly.var(axis=0,ddof=0)

        ls = {
            'ls_x':ls_x, # across-area loading similarity for area 1
            'ls_y':ls_y, # across-area loading similarity for area 2
            'ls_priv_x':ls_priv_x, # within-area loading similarity for area 1
            'ls_priv_y':ls_priv_y  # within-area loading similarity for area 2
        }
        return ls

    def compute_dshared(self,cutoff_thresh=0.95):
        '''
        Compute shared dimensionality (d_shared) in each across- and within-area loading matrix.

                Parameters:
                        cutoff_thresh (float): Cutoff percentage (0-1) of across- or within-area shared variance to explain for selecting d_shared

                Returns:
                        return_dict (dict): Dictionary containing the across- and within-area d_shared for each area
        '''

        Wx,Wy,Lx,Ly = self.get_loading_matrices()

        # for across-area
        if self.params['zDim'] > 0:
            # area 1
            shared_x = Wx.dot(Wx.T)
            s = slin.svdvals(shared_x) # eigenvalues of WWT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_x = dims[0]+1

            # area 2
            shared_y = Wy.dot(Wy.T)
            s = slin.svdvals(shared_y) # eigenvalues of WWT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_y = dims[0]+1

            # overall
            W = np.concatenate((Wx,Wy),axis=0)
            shared = W.dot(W.T)
            s = slin.svdvals(shared) # eigenvalues of WWT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_all = dims[0]+1
        else:
            dshared_x = 0
            dshared_y = 0
            dshared_all = 0

        # for within area 1
        if self.params['zxDim'] > 0:
            shared_x = Lx.dot(Lx.T)
            s = slin.svdvals(shared_x) # eigenvalues of LLT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_priv_x = dims[0]+1
        else:
            dshared_priv_x = 0

        # for within area 2
        if self.params['zyDim'] > 0:
            shared_y = Ly.dot(Ly.T)
            s = slin.svdvals(shared_y) # eigenvalues of LLT
            var_exp = np.cumsum(s)/np.sum(s)
            dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
            dshared_priv_y = dims[0]+1
        else:
            dshared_priv_y = 0

        return_dict = {
            'dshared_x':dshared_x, # d_shared for across-area shared variance in area 1
            'dshared_y':dshared_y, # d_shared for across-area shared variance in area 2
            'dshared_priv_x':dshared_priv_x, # d_shared for within-area shared variance in area 1
            'dshared_priv_y':dshared_priv_y, # d_shared for within-area shared variance in area 2
            'dshared_all':dshared_all # d_shared for across-area shared variance jointly for area 1 and 2
        }

        return return_dict

    def compute_part_ratio(self):
        '''
        Compute part ratio in each across- and within-area loading matrix.

                Returns:
                        return_dict (dict): Dictionary containing the part ratio for each across- and within-area loading matrix
        '''

        Wx,Wy,Lx,Ly = self.get_loading_matrices()

        # for area 1
        shared_x = Wx.dot(Wx.T)
        s = slin.svd(shared_x,full_matrices=False,compute_uv=False)
        pr_x = np.square(s.sum()) / np.square(s).sum()
        
        shared_x = Lx.dot(Lx.T)
        s = slin.svd(shared_x,full_matrices=False,compute_uv=False)
        pr_priv_x = np.square(s.sum()) / np.square(s).sum()

        # for area 2
        shared_y = Wy.dot(Wy.T)
        s = slin.svd(shared_y,full_matrices=False,compute_uv=False)
        pr_y = np.square(s.sum()) / np.square(s).sum()

        shared_y = Ly.dot(Ly.T)
        s = slin.svd(shared_y,full_matrices=False,compute_uv=False)
        pr_priv_y = np.square(s.sum()) / np.square(s).sum()

        # overall
        W = np.concatenate((Wx,Wy),axis=0)
        shared = W.dot(W.T)
        s = slin.svd(shared,full_matrices=False,compute_uv=False)
        pr = np.square(s.sum()) / np.square(s).sum()

        return_dict = {
            'pr':pr, # overall part ratio for across-area (includes area 1 and 2)
            'pr_x':pr_x, # part ratio for across-area loading matrix in area 1
            'pr_y':pr_y, # part ratio for across-area loading matrix in area 2
            'pr_priv_x':pr_priv_x, # part ratio for within-area loading matrix in area 1
            'pr_priv_y':pr_priv_y  # part ratio for within-area loading matrix in area 2
        }

        return return_dict

    def compute_psv(self):
        '''
        Compute percentage of shared variance (%sv) in each across- and within-area loading matrix.

                Returns:
                        psv (dict): Dictionary containing the across- and within-area %sv for neurons in each area 
        '''

        Wx,Wy,Lx,Ly = self.get_loading_matrices()
        priv_x,priv_y = self.params['psi_x'],self.params['psi_y']
        
        shared_acc_x = np.diag(Wx.dot(Wx.T))
        shared_acc_y = np.diag(Wy.dot(Wy.T))
        shared_with_x = np.diag(Lx.dot(Lx.T))
        shared_with_y = np.diag(Ly.dot(Ly.T))
        total_x = shared_acc_x + shared_with_x + priv_x
        total_y = shared_acc_y + shared_with_y + priv_y
        
        # for area 1
        ind_psv_x = (shared_acc_x / total_x).flatten() * 100
        ind_psv_priv_x = (shared_with_x / total_x).flatten() * 100
        priv_var_x = (priv_x / total_x).flatten() * 100
        psv_x = np.mean(ind_psv_x)
        psv_priv_x = np.mean(ind_psv_priv_x)

        # for area 2
        ind_psv_y = (shared_acc_y / total_y).flatten() * 100
        ind_psv_priv_y = (shared_with_y / total_y).flatten() * 100
        priv_var_y = (priv_y / total_y).flatten() * 100
        psv_y = np.mean(ind_psv_y)
        psv_priv_y = np.mean(ind_psv_priv_y)

        # overall
        psv_overall = np.mean(np.concatenate((ind_psv_x,ind_psv_y)))
        psv_priv_overall = np.mean(np.concatenate((ind_psv_priv_x,ind_psv_priv_y)))

        psv = {
            'ind_priv_x':priv_var_x, # percent of independent variance for each neuron in area 1
            'ind_priv_y': priv_var_y, # percent of independent variance for each neuron in area 2
            'ind_psv_x':ind_psv_x, # percent of across-area variance for each neuron in area 1
            'ind_psv_y':ind_psv_y, # percent of across-area variance for each neuron in area 2
            'ind_psv_priv_x':ind_psv_priv_x, # percent of within-area variance for each neuron in area 1
            'ind_psv_priv_y':ind_psv_priv_y, # percent of within-area variance for each neuron in area 2
            'psv_x':psv_x, # percent of across-area variance, averaged across neurons in area 1
            'psv_y':psv_y, # percent of across-area variance, averaged across neurons in area 2
            'psv_priv_x':psv_priv_x, # percent of within-area variance, averaged across neurons in area 1
            'psv_priv_y':psv_priv_y, # percent of within-area variance, averaged across neurons in area 1
            'psv_all':psv_overall, # percent of across-area variance, averaged across all neurons
            'psv_priv_all':psv_priv_overall # percent of within-area variance, averaged across all neurons
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