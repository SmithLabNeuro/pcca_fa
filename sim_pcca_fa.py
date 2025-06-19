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

    def __init__(self,n1,n2,d,d1,d2,rand_seed=None,flat_eigs=False,sv_goal=(25,25),theta=None):
        '''
        Initialize simulator by generating loading matrices and other necessary parameters.

                Parameters:
                    n1 (int): Number of neurons in area 1
                    n2 (int): Number of neurons in area 2
                    d (int): Number of across-area latent variables (shared between area 1 and 2)
                    d1 (int): Number of within-area latent variables (only in area 1)
                    d2 (int): Number of within-area latent variables (only in area 2)
                    rand_seed (int): Seed for random number generator, provide to ensure reproducibility
                    flat_eigs (bool): Whether to use flat distribution of singular values (True) or exponential distribution (False) for across- and within-area loading matrices
                    sv_goal (tuple): Percentages of total variance to attribute to (across,within)-area components
                    theta (float): If not None, angle (degrees) to enforce between the top across- and within-area co-fluctuation patterns

        '''
        self.n1 = n1
        self.n2 = n2
        self.d = d
        self.d1 = d1
        self.d2 = d2

        # set random seed
        rng = np.random.default_rng(seed=rand_seed)
        
        # generate model parameters
        mu_x1 = rng.standard_normal(size=(n1))
        mu_x2 = rng.standard_normal(size=(n2))

        across_scale = sv_goal[0]
        within_scale_x1 = sv_goal[1]
        within_scale_x2 = sv_goal[1]
        if across_scale + sv_goal[1] > 100 or np.any(sv_goal)<0:
            raise ValueError("sv_goal must be 0-100%")
        if d == 0 and across_scale > 0:
            warnings.warn("d is 0, setting across-area sv_goal to 0")
            across_scale = 0
        if d1 == 0 and within_scale_x1 > 0:
            warnings.warn("d1 is 0, setting within-area sv_goal to 0")
            within_scale_x1 = 0
        if d2 == 0 and within_scale_x2 > 0:
            warnings.warn("d2 is 0, setting within-area sv_goal to 0")
            within_scale_x2 = 0

        if flat_eigs:
            # flat distribution
            eig_z = np.ones(d)
            eig_z1 = np.ones(d1)
            eig_z2 = np.ones(d2)
        else:
            # exponential distribution
            eig_z = np.exp(-np.arange(d)/2)
            eig_z1 = np.exp(-np.arange(d1)/2)
            eig_z2 = np.exp(-np.arange(d2)/2)
            
        eig_z = np.diag(np.sqrt(eig_z))
        eig_z1 = np.diag(np.sqrt(eig_z1))
        eig_z2 = np.diag(np.sqrt(eig_z2))

        # genrate loading matrices, then orthonormalize
        W_1 = rng.standard_normal(size=(n1,d))
        W_2 = rng.standard_normal(size=(n2,d))
        L_1 = rng.standard_normal(size=(n1,d1))
        L_2 = rng.standard_normal(size=(n2,d2))

        # orthogonalize for scaling purposes
        if d > 0:
            u,_,_ = slin.svd(W_1,full_matrices=False)
            W_1 = u @ eig_z
            u,_,_ = slin.svd(W_2,full_matrices=False)
            W_2 = u @ eig_z
        if d1 > 0:
            u,_,_ = slin.svd(L_1,full_matrices=False)
            L_1 = u @ eig_z1
        if d2 > 0:
            u,_,_ = slin.svd(L_2,full_matrices=False)
            L_2 = u @ eig_z2

        # if desired to rotate eigenvector of loading matrix, do that now
        if theta is not None:
            # first rotate area 1
            unit_vector = W_1[:,0] / slin.norm(W_1[:,0])
            rotated_vector = self.rotate_by_theta(unit_vector, theta, n1) # generate vector theta degrees from the first col of W
            rest_cols = np.random.randn(n1, d1-1) # generate remaining columns of L_1
            tmp = np.column_stack((rotated_vector, rest_cols))
            u,_ = slin.qr(tmp)
            L_1 = u[:,:d1] @ eig_z1
            # now rotate area 2
            unit_vector = W_2[:,0] / slin.norm(W_2[:,0])
            rotated_vector = self.rotate_by_theta(unit_vector, theta, n2) # generate vector theta degrees from the first col of W
            rest_cols = np.random.randn(n2, d2-1) # generate remaining columns of L_2
            tmp = np.column_stack((rotated_vector, rest_cols))
            u,_ = slin.qr(tmp)
            L_2 = u[:,:d2] @ eig_z2

        # scale within-area loading matrices to achieve sv_goal
        shared_across_x1 = np.diag(W_1 @ W_1.T)
        shared_within_x1 = np.diag(L_1 @ L_1.T)
        if d > 0 and d1 > 0:
            curr_scale = shared_across_x1.mean() / shared_within_x1.mean()
            desired_scale = across_scale / within_scale_x1
            multiplier = np.sqrt(curr_scale / desired_scale)
            u,_ = slin.qr(L_1) # get orthonormal basis for L_1 to scale (keep first column the same in case of rotation)
            L_1 = u[:,:d1] @ (eig_z1 * multiplier)
            shared_within_x1 = np.diag(L_1 @ L_1.T)

        shared_across_x2 = np.diag(W_2 @ W_2.T)
        shared_within_x2 = np.diag(L_2 @ L_2.T)
        if d > 0 and d2 > 0:
            curr_scale = shared_across_x2.mean() / shared_within_x2.mean()
            desired_scale = across_scale / within_scale_x2
            multiplier = np.sqrt(curr_scale / desired_scale)
            u,_ = slin.qr(L_2) # get orthonormal basis for L_2 to scale (keep first column the same in case of rotation)
            L_2 = u[:,:d2] @ (eig_z2 * multiplier)
            shared_within_x2 = np.diag(L_2 @ L_2.T)

        # compute independent variance values to maintain total level of shared variance
        if d ==0 and d1 == 0:
            psi_1 = np.ones(n1)
        else:
            snr = (across_scale+within_scale_x1) / (100-across_scale-within_scale_x1)
            psi_1 = (shared_across_x1 + shared_within_x1) / snr
        
        if d ==0 and d2 == 0:
            psi_2 = np.ones(n2)
        else:
            snr = (across_scale+within_scale_x2) / (100-across_scale-within_scale_x2)
            psi_2 = (shared_across_x2 + shared_within_x2) / snr

        L_top = np.concatenate((W_1,L_1,np.zeros((n1,d2))),axis=1)
        L_bottom = np.concatenate((W_2,np.zeros((n2,d1)),L_2),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)

        # store model parameters in dict
        params = {
            'mu_x1':mu_x1,'mu_x2':mu_x2,
            'L_total': L_total,
            'W_1':W_1,'W_2':W_2,
            'L_1':L_1,'L_2':L_2,
            'psi_1':psi_1,'psi_2':psi_2,
            'd':d,'d1':d1,'d2':d2,
        }
        self.params = params


    def sim_data(self,N,rand_seed=None):
        '''
        Simulate data according to pCCA-FA model.

                Parameters:
                        N (int): Number of trials to generate
                        rand_seed (int): Seed for random number generator, provide to ensure reproducibility

                Returns:
                        X_1 (array): Array of size N x n1, simulated activity for area 1
                        X_2 (array): Array of size N x n2, simulated activity for area 2
        '''
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        mu_x1 = self.params['mu_x1'].reshape(self.n1,1)
        mu_x2 = self.params['mu_x2'].reshape(self.n2,1)
        W_1,W_2 = self.params['W_1'], self.params['W_2']
        L_1,L_2 = self.params['L_1'], self.params['L_2']
        psi_1,psi_2 = self.params['psi_1'], self.params['psi_2']

        # draw from each latent variable
        z = np.random.randn(self.d,N)
        z1 = np.random.randn(self.d1,N)
        z2 = np.random.randn(self.d2,N)

        # generate data
        ns_x1 = np.diag(np.sqrt(psi_1)).dot(np.random.randn(self.n1,N))
        ns_x2 = np.diag(np.sqrt(psi_2)).dot(np.random.randn(self.n2,N))
        X_1 = (W_1.dot(z) + L_1.dot(z1) + ns_x1) + mu_x1
        X_2 = (W_2.dot(z) + L_2.dot(z2) + ns_x2) + mu_x2

        return X_1.T, X_2.T
    
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
        self.n1 = params['W_1'].shape[0]
        self.n2 = params['W_2'].shape[0]
        self.d = params['d']
        self.d1 = params['d1']
        self.d2 = params['d2']
        self.params = params

    def rotate_by_theta(self,vec,theta,n):
        '''
        Generate vector that is rotated a specified angle from the given vector.

                Parameters:
                        vec (1D vector): Basis vector about which to perform rotation
                        theta (float): Angle in degrees by which to rotate the given vector
                        n (int): Number of neurons in the area in which rotation is being applied (can pertain to either area 1 or 2)
                
                Returns:
                        rotated_vec (1D vector): The rotated vector
        '''
        # define low-d basis and project vec into plane
        A = np.column_stack((vec, np.random.randn(n,1)))
        q,_ = np.linalg.qr(A)
        v1 = np.transpose(q) @ (vec) # 2 x 1

        # rotation matrix (R)
        theta_rad = math.radians(theta)
        cos_theta, sin_theta = math.cos(theta_rad), math.sin(theta_rad)
        R = np.array([[cos_theta, sin_theta],
                      [-sin_theta, cos_theta]]) 
        
        # rotate vec1 to get desired vec2, then project back to high-d
        vec2_2d = R @ v1 # 2 x 1
        rotated_vec = q @ vec2_2d # n x 1

        return rotated_vec
    
    def orthogonalize(self,do_across=True):
        '''
        Orthogonalize across- (optional) and within-area loading matrices using singular value decomposition. 

                Parameters:
                        do_across (bool): Whether to orthogonalize the across-area loading matrices (True) or not (False)
                
        '''
        # orthogonalize loading matrices
        if do_across:
            u,s,_ = slin.svd(self.params['W_1'],full_matrices=False)
            W_1 = u @ np.diag(s)
            u,s,_ = slin.svd(self.params['W_2'],full_matrices=False)
            W_2 = u @ np.diag(s)
            self.params['W_1'] = W_1
            self.params['W_2'] = W_2
        u,s,_ = slin.svd(self.params['L_1'],full_matrices=False)
        L_1 = u @ np.diag(s)
        u,s,_ = slin.svd(self.params['L_2'],full_matrices=False)
        L_2 = u @ np.diag(s)
        self.params['L_1'] = L_1
        self.params['L_2'] = L_2

        L_top = np.concatenate((W_1,L_1,np.zeros((self.n1,self.d2))),axis=1)
        L_bottom = np.concatenate((W_2,np.zeros((self.n2,self.d1)),L_2),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        self.params['L_total'] = L_total

    def apply_rotation(self,theta,hem='1'):
        '''
        Apply rotation by theta degrees to the first column of the within-area loading matrix of the specified area.

                Parameters:
                        theta (float): Angle in degrees by which to rotate
                        hem (str): Must be one of '1' or '2', indicates the area in which to perform rotation

        '''        
        # first, orthogonalize the loading matrices
        self.orthogonalize(do_across=True)
        params = self.params.copy()

        # rotate eigenvector of loading matrix
        if hem == '1':
            # perform rotation in area X
            s = slin.svdvals(params['L_1'])
            col1 = self.rotate_by_theta(params['W_1'][:,0], theta, self.n1) # generate vector theta degrees from the first col of W_1
            # generate remaining columns of L_1
            rest_cols = np.random.randn(self.n1, self.d1-1)
            tmp = np.column_stack((col1, rest_cols))
            u,_ = slin.qr(tmp)
            L_1 = u[:,:self.d1] @ np.diag(s[:self.d1])
            self.params['L_1'] = L_1
        elif hem == '2':
            # perform rotation in area Y
            s = slin.svdvals(params['L_2'])
            col1 = self.rotate_by_theta(params['W_2'][:,0], theta, self.n2) # generate vector theta degrees from the first col of W_2
            # generate remaining columns of L_1
            rest_cols = np.random.randn(self.n2, self.d2-1)
            tmp = np.column_stack((col1, rest_cols))
            u,_ = slin.qr(tmp)
            L_2 = u[:,:self.d2] @ np.diag(s[:self.d2])
            self.params['L_2'] = L_2
        else:
            return ValueError("hem must be '1' or '2'")
        
        L_top = np.concatenate((params['W_1'],self.params['L_1'],np.zeros((self.n1,self.d2))),axis=1)
        L_bottom = np.concatenate((params['W_2'],np.zeros((self.n2,self.d1)),self.params['L_2']),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        self.params['L_total'] = L_total