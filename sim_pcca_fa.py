import numpy as np
import scipy.linalg as slin
import warnings, math

class sim_pcca_fa:
    '''
    Class to store parameters for generating simulated data according to the pCCA-FA model.

    Methods
    -------
    sim_data()
        Simulate data according to pCCA-FA model.
    get_params()
        Get parameters of simulator.
    set_params()
        Set parameters of simulator.
    rotate_by_theta()
        Generate vector that is rotated a specified angle from the given vector.
    orthogonalize()
        Orthogonalize across- (optional) and within-area loading matrices using singular value decomposition. 
    apply_rotation()
        Apply rotation by theta degrees to the first column of the within-area loading matrix of the specified area.
    '''

    def __init__(self,xDim,yDim,zDim,zxDim,zyDim,rand_seed=None,flat_eigs=False,sv_goal=(25,25),theta=None):
        '''
        Initialize simulator by generating loading matrices and other necessary parameters.

                Parameters:
                    xDim (int): Number of neurons in area X
                    yDim (int): Number of neurons in area Y
                    zDim (int): Number of across-area latent variables (shared between area X and Y)
                    zxDim (int): Number of within-area latent variables (only in area X)
                    zyDim (int): Number of within-area latent variables (only in area Y)
                    rand_seed (int): Seed for random number generator, provide to ensure reproducibility
                    flat_eigs (bool): Whether to use flat eigenspectra (True) or exponential eigenspectra (False) for across- and within-area loading matrices
                    sv_goal (tuple): Percentages of total variance to attribute to (across,within)-area components
                    theta (float): If not None, angle (degrees) to enforce between the top across- and within-area co-fluctuation patterns

        '''
        self.xDim = xDim
        self.yDim = yDim
        self.zDim = zDim
        self.zxDim = zxDim
        self.zyDim = zyDim

        # set random seed
        rng = np.random.default_rng(seed=rand_seed)
        
        # generate model parameters
        mu_x = rng.standard_normal(size=(xDim))
        mu_y = rng.standard_normal(size=(yDim))

        across_scale = sv_goal[0]
        within_scale_x = sv_goal[1]
        within_scale_y = sv_goal[1]
        if across_scale + sv_goal[1] > 100 or np.any(sv_goal)<0:
            raise ValueError("sv_goal must be 0-100%")
        if zDim == 0 and across_scale > 0:
            warnings.warn("zDim is 0, setting across-area sv_goal to 0")
            across_scale = 0
        if zxDim == 0 and within_scale_x > 0:
            warnings.warn("zxDim is 0, setting within-area sv_goal to 0")
            within_scale_x = 0
        if zyDim == 0 and within_scale_y > 0:
            warnings.warn("zyDim is 0, setting within-area sv_goal to 0")
            within_scale_y = 0

        if flat_eigs:
            # flat distribution
            eig_z = np.ones(zDim)
            eig_zx = np.ones(zxDim)
            eig_zy = np.ones(zyDim)
        else:
            # exponential distribution
            eig_z = np.exp(-np.arange(zDim)/2)
            eig_zx = np.exp(-np.arange(zxDim)/2)
            eig_zy = np.exp(-np.arange(zyDim)/2)
            
        eig_z = np.diag(np.sqrt(eig_z))
        eig_zx = np.diag(np.sqrt(eig_zx))
        eig_zy = np.diag(np.sqrt(eig_zy))

        # genrate loading matrices, then orthonormalize
        W_x = rng.standard_normal(size=(xDim,zDim))
        W_y = rng.standard_normal(size=(yDim,zDim))
        L_x = rng.standard_normal(size=(xDim,zxDim))
        L_y = rng.standard_normal(size=(yDim,zyDim))

        # orthogonalize for scaling purposes
        if zDim > 0:
            uwx,_,_ = slin.svd(W_x)
            W_x = uwx[:,:zDim] @ eig_z
            uwy,_,_ = slin.svd(W_y)
            W_y = uwy[:,:zDim] @ eig_z
        if zxDim > 0:
            ulx,_,_ = slin.svd(L_x)
            L_x = ulx[:,:zxDim] @ eig_zx
        if zyDim > 0:
            uly,_,_ = slin.svd(L_y)
            L_y = uly[:,:zyDim] @ eig_zy

        # if desired to rotate eigenvector of loading matrix, do that now
        if theta is not None:
            # first rotate area X
            col1 = self.rotate_by_theta(uwx[:,0], theta, xDim) # generate vector theta degrees from the first col of W
            rest_cols = np.random.randn(xDim, zxDim-1) # generate remaining columns of L_x
            tmp = np.column_stack((col1, rest_cols))
            ulx,_ = slin.qr(tmp)
            L_x = ulx[:,:zxDim] @ eig_zx
            # now rotate area Y
            col1 = self.rotate_by_theta(uwy[:,0], theta, yDim) # generate vector theta degrees from the first col of W
            rest_cols = np.random.randn(yDim, zyDim-1) # generate remaining columns of L_y
            tmp = np.column_stack((col1, rest_cols))
            uly,_ = slin.qr(tmp)
            L_y = uly[:,:zyDim] @ eig_zy

        # scale within-area loading matrices to achieve sv_goal
        shared_across_x = np.diag(W_x @ W_x.T)
        shared_within_x = np.diag(L_x @ L_x.T)
        if zDim > 0 and zxDim > 0:
            curr_scale = shared_across_x.mean() / shared_within_x.mean()
            desired_scale = across_scale / within_scale_x
            multiplier = np.sqrt(curr_scale / desired_scale)
            L_x = ulx[:,:zxDim] @ (eig_zx * multiplier)
            shared_within_x = np.diag(L_x @ L_x.T)

        shared_across_y = np.diag(W_y @ W_y.T)
        shared_within_y = np.diag(L_y @ L_y.T)
        if zDim > 0 and zyDim > 0:
            curr_scale = shared_across_y.mean() / shared_within_y.mean()
            desired_scale = across_scale / within_scale_y
            multiplier = np.sqrt(curr_scale / desired_scale)
            L_y = uly[:,:zyDim] @ (eig_zy * multiplier)
            shared_within_y = np.diag(L_y @ L_y.T)

        # compute independent variance values to maintain total level of shared variance
        if zDim ==0 and zxDim == 0:
            psi_x = np.ones(xDim)
        else:
            snr = (across_scale+within_scale_x) / (100-across_scale-within_scale_x)
            psi_x = (shared_across_x + shared_within_x) / snr
        
        if zDim ==0 and zyDim == 0:
            psi_y = np.ones(yDim)
        else:
            snr = (across_scale+within_scale_y) / (100-across_scale-within_scale_y)
            psi_y = (shared_across_y + shared_within_y) / snr

        L_top = np.concatenate((W_x,L_x,np.zeros((xDim,zyDim))),axis=1)
        L_bottom = np.concatenate((W_y,np.zeros((yDim,zxDim)),L_y),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)

        # store model parameters in dict
        params = {
            'mu_x':mu_x,'mu_y':mu_y,
            'L_total': L_total,
            'W_x':W_x,'W_y':W_y,
            'L_x':L_x,'L_y':L_y,
            'psi_x':psi_x,'psi_y':psi_y,
            'zDim':zDim,'zxDim':zxDim,'zyDim':zyDim,
        }
        self.params = params


    def sim_data(self,N,rand_seed=None):
        '''
        Simulate data according to pCCA-FA model.

                Parameters:
                        N (int): Number of trials to generate
                        rand_seed (int): Seed for random number generator, provide to ensure reproducibility

                Returns:
                        X (array): Array of size N x xDim, simulated activity for area X
                        Y (array): Array of size N x yDim, simulated activity for area Y
        '''
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        mu_x = self.params['mu_x'].reshape(self.xDim,1)
        mu_y = self.params['mu_y'].reshape(self.yDim,1)
        W_x,W_y = self.params['W_x'], self.params['W_y']
        L_x,L_y = self.params['L_x'], self.params['L_y']
        psi_x,psi_y = self.params['psi_x'], self.params['psi_y']

        # draw from each latent variable
        z = np.random.randn(self.zDim,N)
        zx = np.random.randn(self.zxDim,N)
        zy = np.random.randn(self.zyDim,N)

        # generate data
        ns_x = np.diag(np.sqrt(psi_x)).dot(np.random.randn(self.xDim,N))
        ns_y = np.diag(np.sqrt(psi_y)).dot(np.random.randn(self.yDim,N))
        X = (W_x.dot(z) + L_x.dot(zx) + ns_x) + mu_x
        Y = (W_y.dot(z) + L_y.dot(zy) + ns_y) + mu_y

        return X.T, Y.T
    
    def get_params(self):
        '''
        Get parameters of simulator.

                Returns:
                        params (dict): Dictionary containing each parameter of the simulator
        '''
        return self.params

    def set_params(self,params):
        '''
        Set parameters of simulator.

                Parameters:
                        params (dict): Dictionary containing each parameter of the simulator
        '''
        # determine dimensionalities
        self.xDim = params['W_x'].shape[0]
        self.yDim = params['W_y'].shape[0]
        self.zDim = params['zDim']
        self.zxDim = params['zxDim']
        self.zyDim = params['zyDim']
        self.params = params

    def rotate_by_theta(self,vec,theta,xDim):
        '''
        Generate vector that is rotated a specified angle from the given vector.

                Parameters:
                        vec (1D vector): Basis vector about which to perform rotation
                        theta (float): Angle in degrees by which to rotate the given vector
                        xDim (int): Number of neurons in the area in which rotation is being applied (can pertain to either area X or Y)
                
                Returns:
                        rotated_vec (1D vector): The rotated vector
        '''
        # define low-d basis and project vec into plane
        A = np.column_stack((vec, np.random.randn(xDim,1)))
        q,_ = np.linalg.qr(A)
        v1 = np.transpose(q) @ (vec) # 2 x 1

        # rotation matrix (R)
        theta_rad = math.radians(theta)
        cos_theta, sin_theta = math.cos(theta_rad), math.sin(theta_rad)
        R = np.array([[cos_theta, sin_theta],
                      [-sin_theta, cos_theta]]) 
        
        # rotate vec1 to get desired vec2, then project back to high-d
        vec2_2d = R @ v1 # 2 x 1
        rotated_vec = q @ vec2_2d # xDim x 1

        return rotated_vec
    
    def orthogonalize(self,do_across=True):
        '''
        Orthogonalize across- (optional) and within-area loading matrices using singular value decomposition. 

                Parameters:
                        do_across (bool): Whether to orthogonalize the across-area loading matrices (True) or not (False)
                
        '''
        # orthogonalize loading matrices
        if do_across:
            uwx,svwx,_ = slin.svd(self.params['W_x'])
            W_x = uwx[:,:self.zDim] @ np.diag((svwx[:self.zDim]))
            uwy,svwy,_ = slin.svd(self.params['W_y'])
            W_y = uwy[:,:self.zDim] @ np.diag((svwy[:self.zDim]))
            self.params['W_x'] = W_x
            self.params['W_y'] = W_y
        ulx,svlx,_ = slin.svd(self.params['L_x'])
        L_x = ulx[:,:self.zxDim] @ np.diag(svlx[:self.zxDim])
        uly,svly,_ = slin.svd(self.params['L_y'])
        L_y = uly[:,:self.zyDim] @ np.diag(svly[:self.zyDim])
        self.params['L_x'] = L_x
        self.params['L_y'] = L_y

        L_top = np.concatenate((W_x,L_x,np.zeros((self.xDim,self.zyDim))),axis=1)
        L_bottom = np.concatenate((W_y,np.zeros((self.yDim,self.zxDim)),L_y),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        self.params['L_total'] = L_total

    def apply_rotation(self,theta,hem='x'):
        '''
        Apply rotation by theta degrees to the first column of the within-area loading matrix of the specified area.

                Parameters:
                        theta (float): Angle in degrees by which to rotate
                        hem (str): Must be one of 'x' or 'y', indicates the area in which to perform rotation

        '''        
        # first, orthogonalize the loading matrices
        self.orthogonalize(do_across=True)
        params = self.params.copy()

        # rotate eigenvector of loading matrix
        if hem == 'x':
            # perform rotation in area X
            svlx = slin.svdvals(params['L_x'])
            col1 = self.rotate_by_theta(params['W_x'][:,0], theta, self.xDim) # generate vector theta degrees from the first col of W
            # generate remaining columns of L_x
            rest_cols = np.random.randn(self.xDim, self.zxDim-1)
            tmp = np.column_stack((col1, rest_cols))
            ulx,_ = slin.qr(tmp)
            L_x = ulx[:,:self.zxDim] @ np.diag(svlx[:self.zxDim])
            self.params['L_x'] = L_x
        elif hem == 'y':
            # perform rotation in area Y
            svly = slin.svdvals(params['L_y'])
            col1 = self.rotate_by_theta(params['W_y'][:,0], theta, self.yDim) # generate vector theta degrees from the first col of W
            # generate remaining columns of L_x
            rest_cols = np.random.randn(self.yDim, self.zyDim-1)
            tmp = np.column_stack((col1, rest_cols))
            uly,_ = slin.qr(tmp)
            L_y = uly[:,:self.zyDim] @ np.diag(svly[:self.zyDim])
            self.params['L_y'] = L_y
        else:
            return ValueError("hem must be 'x' or 'y'")
        
        L_top = np.concatenate((params['W_x'],self.params['L_x'],np.zeros((self.xDim,self.zyDim))),axis=1)
        L_bottom = np.concatenate((params['W_y'],np.zeros((self.yDim,self.zxDim)),self.params['L_y']),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        self.params['L_total'] = L_total