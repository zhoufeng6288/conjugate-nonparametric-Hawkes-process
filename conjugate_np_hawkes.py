import numpy as np
import copy
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import uniform
from scipy.special import expit
from numpy.polynomial import legendre
from scipy.special import psi



class CNPHawkes:
    """
    This class implements the inference for nonparametric Hawkes process with auxiliary latent variables.
    The main features are three statistical inference methods: Gibbs sampler, EM algorithm and mean-field variational inference. 
    """
    def __init__(self, ind_p_mu, ind_p_phi, T_phi):
        """
        Initialises an instance and set its hyperparameters.

        :type ind_p_mu: numpy array
        :param ind_p_mu: the inducing points for baseline intensity
        :type ind_p_phi: numpy array
        :param ind_p_phi: the inducing points for influence function
        :type T_phi: float
        :param T_phi: the support of influence function
        """
        self.ind_p_mu = ind_p_mu
        self.ind_p_phi = ind_p_phi
        self.T_phi = T_phi

        self.theta0_mu = 0 
        self.theta1_mu = 0
        self.noise_var_mu = 0
        self.theta0_phi = 0
        self.theta1_phi = 0
        self.noise_var_phi = 0

        self.points_hawkes = None
        self.T = 0
        self.points_hawkes_test = None
        self.T_test = 0

    def set_kernel_hyperparameters(self, theta0_mu, theta1_mu, noise_var_mu, theta0_phi, theta1_phi, noise_var_phi):
        r"""
        Fix the hyperparameters of GP kernel for baseline intensity and influence function.
        Here we assume it is RBF kernel :math:`K(x,x')=\theta0*exp(-theta1/2*(||x - x'||**2)) + \sigma^2*I`. 

        :type theta0_mu: float
        :param theta0_mu: theta0 for baseline intensity
        :type theta1_mu: float
        :param theta1_mu: theta1 for baseline intensity
        :type noise_var_mu: float
        :param noise_var_mu: noise for baseline intensity
        :type theta0_phi: float
        :param theta0_phi: theta0 for influence function
        :type theta1_phi: float
        :param theta1_phi: theta1 for influence function
        :type noise_var_phi: float
        :param noise_var_phi: noise for influence function
        """
        self.theta0_mu = theta0_mu
        self.theta1_mu = theta1_mu
        self.noise_var_mu = noise_var_mu
        self.theta0_phi = theta0_phi
        self.theta1_phi = theta1_phi
        self.noise_var_phi = noise_var_phi

    def set_train_test_data(self, points_hawkes, T, points_hawkes_test, T_test):
        r"""
        Set the training/test data and observation window for Hawkes processes. 

        :type points_hawkes: list
        :param points_hawkes: the training points
        :type T: float
        :param T: the observation window [0,T] for training points
        :type points_hawkes_test: list
        :param points_hawkes_test: the test points
        :type T_test: float
        :param T_test: the observation window [0,T_test] for test points
        """
        self.points_hawkes = points_hawkes
        self.T = T
        self.points_hawkes_test = points_hawkes_test
        self.T_test = T_test

#########################################################################################################

    'tool function'
    @staticmethod
    def intensity_discrete_phi_mu(t, history, mu, phi, T_phi, T):
        r"""
        Compute the intensity at t given the historical timestamps.

        :type t: float
        :param t: the target time
        :type history: list
        :param history: the timestamps before t
        :type mu: float
        :param mu: the baseline intensity
        :type phi: numpy array
        :param phi: the influence function
        :type T_phi: float
        :param T_phi: the support of influence function
        :type T: float
        :param T: the observation window
        :rtype: float
        :return: the intensity at t
        """
        N=len(phi)
        M=len(mu)
        intensity=0
        for i in range(len(history)):
            if history[i]>=t:
                break
            delta_t=t-history[i]
            if delta_t<T_phi:
                intensity+=phi[int(delta_t*N/T_phi)]
        intensity=mu[int(t/T*M)]+intensity
        return intensity

    def loglikelihood_discrete_phi_mu(self, points_hawkes, mu, phi, T_phi, T):
        r"""
        Compute the loglikelihood of observation using given :math:`\mu` and :math:`\phi`.

        :type points_hawkes: list
        :param points_hawkes: the observed timestamps
        :type mu: float
        :param mu: the baseline intensity
        :type phi: numpy array
        :param phi: the influence function
        :type T_phi: float
        :param T_phi: the support of influence function
        :type T: float
        :param T: the observation window
        :rtype: float
        :return: the loglikelihood
        """
        N=len(points_hawkes)
        M=len(phi)
        logl=0
        for i in range(N):
            logl += np.log(self.intensity_discrete_phi_mu(points_hawkes[i],points_hawkes,mu, phi, T_phi, T))
            delta_t = T - points_hawkes[i]
            if delta_t >= T_phi:
                logl -= sum(phi*T_phi/M)
            else:
                temp = delta_t*M/T_phi
                temp_int = int(temp)
                logl -= sum(phi[:temp_int]*T_phi/M)+phi[temp_int]*T_phi/M*(temp-temp_int)
        return logl-sum(mu*T/len(mu))

    @staticmethod
    def rbf_kernel(theta0, theta1, noise_var, x, y): 
        r"""
        Compute the kernel matrix for location vector x, :math:`K(x,x')=\theta0*exp(-theta1/2*(||x - x'||**2)) + \sigma^2*I`. 

        :type theta0: float
        :param theta0: 
        :type theta1: float
        :param theta1: 
        :type noise_var: float 
        :param noise_var: 
        :type x: numpy array
        :param x: the location points
        :type y: numpy array
        :param y: the location points

        :rtype: 2D numpy array
        :return: kernel matrix
        """
        N_x = len(x)
        N_y = len(y)
        a = x.reshape(N_x,1)
        b = y.reshape(1,N_y)
        if N_x == N_y:
            return theta0*np.exp(-theta1/2*(a-b)**2)+np.eye(N_x)*noise_var
        else:
            return theta0*np.exp(-theta1/2*(a-b)**2)

#########################################################################################################

    'Gibbs Sampler'
    @staticmethod
    def PG(b,c):
        r"""
        Sampling from a Polya-Gamma density by truncation (default 2000 samples). It is not efficient.
        
        :type b: float
        :param b: parameter of Polya-Gamma density
        :type c: float
        :param c: parameter of Polya-Gamma density
        :rtype: float
        :return: a sample from Polya-Gamma density
        """
        g=gamma.rvs(b,size=2000)
        d=np.array(range(1,2001))
        d=(d-0.5)**2+c**2/4/np.pi/np.pi
        return sum(g/d)/2/np.pi/np.pi

    def ini_X(self):
        r"""
        Initialize the branching matrix. 

        :rtype: 2D numpy array
        :return: Initial branching matrix.
        """
        N = len(self.points_hawkes)
        X = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1):
                tij = self.points_hawkes[i] - self.points_hawkes[j]
                if tij >= self.T_phi: continue
                else:
                    X[i][j:i+1] = multinomial.rvs(n=1,p=[1/(i+1-j)]*(i+1-j))
                    break
        return X

    def extract_t_tau(self, X):
        r"""
        Extract the endogenous 'tau' and exogenous 't' according to the branching matrix. 

        :type X: 2D numpy array
        :param X: branching matrix
        :rtype: list
        :return: list of t, list of tau
        """
        N = len(self.points_hawkes)
        points_mu=[]
        points_phi=[]
        for i in range(N):
            if X[i][i]==1:
                points_mu.append(self.points_hawkes[i])
        for i in range(1,N):
            for j in range(i):
                if X[i][j]==1:
                    points_phi.append(self.points_hawkes[i] - self.points_hawkes[j])
                    break
        return points_mu, points_phi

    @staticmethod
    def inhomo_simulation(intensity, T):
        r"""
        Simulate an inhomogeneous Poisson process using discrete intensity

        :type intensity: numpy array
        :param intensity: discrete intensity function
        :type T: float
        :param T: time window

        :rtype: list
        :return: inhomogeneous Poisson process simulation
        """
        intensity_ub = np.max(intensity)
        delta_t = T/len(intensity)
        t=0
        points_inhomo=[]
        while(t < T):
            r = expon.rvs(scale = 1/intensity_ub)
            t += r
            if t >= T: break
            D = uniform.rvs(loc=0, scale=1)
            assert intensity[int(t/delta_t)] <= intensity_ub
            if D*intensity_ub <= intensity[int(t/delta_t)]:
                points_inhomo.append(t)
        if points_inhomo!=[] and points_inhomo[-1]>T:
            del points_inhomo[-1]
        return points_inhomo

    def Gibbs(self, num_pre_mu, num_pre_phi, num_iter):
        r"""
        Gibbs sampler which is used to sample from the posterior of lamda_ub_mu, lamda_ub_phi, g_s_mu and g_s_phi. 
        
        :type num_pre_mu: int
        :param num_pre_mu: the number of prediction points on [0,T]
        :type num_pre_phi: int
        :param num_pre_phi: the number of prediction points on [0,T_phi]
        :type num_iter: int
        :param num_iter: the number of Gibbs loops

        :rtype: numpy array
        :return: the posterior samples of \mu(t), \phi(\tau), lamda_ub_mu, lamda_ub_phi, the training and 
        test log-likelihood along Gibbs loops. 
        """
        N = len(self.points_hawkes)
        S_mu = len(self.ind_p_mu)
        S_phi = len(self.ind_p_phi)
        
        # initial X and points
        X = self.ini_X()
        P = np.zeros((N,N))
        points_mu, points_phi = self.extract_t_tau(X)
        w_ii = [self.PG(1,0) for i in range(len(points_mu))]
        w_ij = [self.PG(1,0) for i in range(len(points_phi))]
        
        # initial lamda
        lamda_mu = len(points_mu)*2/self.T
        lamda_phi = len(points_phi)*12/N/self.T_phi
        
        x_M_mu = self.inhomo_simulation(np.array([lamda_mu]), self.T)
        w_M_mu = [self.PG(1,0) for i in range(len(x_M_mu))]
        x_M_phi_flat = []
        w_M_phi_flat = []
        for i in range(N):
            x_M_temp = self.inhomo_simulation(np.array([lamda_phi]), self.T_phi)
            w_M_temp = [self.PG(1,0) for i in range(len(x_M_temp))]
            x_M_phi_flat += x_M_temp
            w_M_phi_flat += w_M_temp
        g_s_mu = np.zeros(S_mu)
        g_s_phi = np.zeros(S_phi)
        g_mu_pos_mean = np.zeros(num_pre_mu) # predictive g_mu function on [0,T]
        g_phi_pos_mean = np.zeros(num_pre_phi) # predictive g_phi function on [0,T_phi]
        
        K_S_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, self.noise_var_mu, self.ind_p_mu, self.ind_p_mu)
        K_S_mu_inv = np.linalg.inv(K_S_mu)
        
        K_S_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi, self.noise_var_phi, self.ind_p_phi, self.ind_p_phi)
        K_S_phi_inv = np.linalg.inv(K_S_phi)
        
        x1_mu = np.linspace(0, self.T, num_pre_mu)      # prediction points
        delta_t_mu = self.T/num_pre_mu
        k_matrix_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, 0, x1_mu, self.ind_p_mu)
        k_K_mu = np.dot(k_matrix_mu, K_S_mu_inv)
        
        x1_phi = np.linspace(0, self.T_phi, num_pre_phi)      # prediction points
        delta_t_phi = self.T_phi/num_pre_phi
        k_matrix_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi, 0, x1_phi, self.ind_p_phi)
        k_K_phi = np.dot(k_matrix_phi, K_S_phi_inv)
        
        mu_list=[]
        phi_list=[]
        lamda_mu_list=[]
        lamda_phi_list=[]
        logl_train_list=[]
        logl_test_list=[]
        
        for iteration in range(num_iter):
            # extract points_mu and points_phi
            points_mu, points_phi = self.extract_t_tau(X)
            
            # sample w_ii and w_ij
            w_ii=[]
            for i in range(len(points_mu)):
                w_ii.append(self.PG(1,g_mu_pos_mean[int(points_mu[i]/delta_t_mu)]))
            w_ij=[]
            for i in range(len(points_phi)):
                w_ij.append(self.PG(1,g_phi_pos_mean[int(points_phi[i]/delta_t_phi)]))
            
            # sample x_M_mu, w_M_mu, x_M_phi_flat, w_M_phi_flat
            x_M_mu=self.inhomo_simulation(lamda_mu*expit(-g_mu_pos_mean), self.T)
            w_M_mu=[self.PG(1,g_mu_pos_mean[int(x_M_mu[i]/delta_t_mu)]) for i in range(len(x_M_mu))]
            
            x_M_phi_flat=[]
            w_M_phi_flat=[]
            for i in range(N):
                x_M_phi_temp = self.inhomo_simulation(lamda_phi*expit(-g_phi_pos_mean), self.T_phi)
                w_M_phi_temp = [self.PG(1,g_phi_pos_mean[int(x_M_phi_temp[i]/delta_t_phi)]) for i in range(len(x_M_phi_temp))]
                x_M_phi_flat += x_M_phi_temp
                w_M_phi_flat += w_M_phi_temp
            
            # sample lamda_mu and lamda_phi
            lamda_mu = gamma(a=len(points_mu)+len(x_M_mu), scale=1/self.T).rvs()
            lamda_phi = gamma(a=len(points_phi)+len(x_M_phi_flat), scale=1/N/self.T_phi).rvs()

            # sample g_s_mu and g_s_phi
            D_mu = np.diag(np.array(w_ii+w_M_mu))
            x_NM_mu = points_mu + x_M_mu
            NM_mu = len(x_NM_mu)
            K_S_NM_mu = self.rbf_kernel(self.theta0_mu,self.theta1_mu,0,self.ind_p_mu,np.array(x_NM_mu))
            cov_S_mu = np.linalg.inv(np.dot(np.dot(np.dot(np.dot(K_S_mu_inv,K_S_NM_mu),D_mu),K_S_NM_mu.T),K_S_mu_inv)+K_S_mu_inv)
            mean_S_mu = np.dot(np.dot(np.dot(cov_S_mu,K_S_mu_inv),K_S_NM_mu),np.array([0.5]*len(points_mu)+[-0.5]*len(x_M_mu)))
            g_s_mu = multivariate_normal(mean_S_mu,cov_S_mu).rvs()
            g_mu_pos_mean = np.dot(k_K_mu, g_s_mu)
            
            D_phi = np.diag(np.array(w_ij+w_M_phi_flat))
            x_NM_phi = points_phi+x_M_phi_flat
            NM_phi = len(x_NM_phi)
            K_S_NM_phi = self.rbf_kernel(self.theta0_phi,self.theta1_phi,0,self.ind_p_phi,np.array(x_NM_phi))
            cov_S_phi=np.linalg.inv(np.dot(np.dot(np.dot(np.dot(K_S_phi_inv,K_S_NM_phi),D_phi),K_S_NM_phi.T),K_S_phi_inv)+K_S_phi_inv)
            mean_S_phi=np.dot(np.dot(np.dot(cov_S_phi,K_S_phi_inv),K_S_NM_phi),np.array([0.5]*len(points_phi)+[-0.5]*len(x_M_phi_flat)))
            g_s_phi=multivariate_normal(mean_S_phi,cov_S_phi).rvs()
            g_phi_pos_mean=np.dot(k_K_phi,g_s_phi)
            
            # sample X
            mu = lamda_mu*expit(g_mu_pos_mean)
            phi = lamda_phi*expit(g_phi_pos_mean)
            
            for i in range(N): # updata of P
                mu_ti = mu[int(self.points_hawkes[i]/delta_t_mu)]
                intensity_total = 0
                for j in range(i):
                    tji = self.points_hawkes[i] - self.points_hawkes[j]
                    if tji >= self.T_phi: continue
                    intensity_total += phi[int(tji/delta_t_phi)]
                intensity_total += mu_ti
                P[i][i] = mu_ti/intensity_total
                for j in range(i):
                    tji = self.points_hawkes[i] - self.points_hawkes[j]
                    if tji >= self.T_phi: P[i][j] = 0
                    else: P[i][j] = phi[int(tji/delta_t_phi)]/intensity_total
                X[i][:(i+1)] = multinomial(n=1,p=P[i][:(i+1)]).rvs()
            assert np.sum(X) == N
            
            # loglikelihood
            logl_train = self.loglikelihood_discrete_phi_mu(self.points_hawkes, mu, phi, self.T_phi, self.T)
            logl_test = self.loglikelihood_discrete_phi_mu(self.points_hawkes_test, mu, phi, self.T_phi, self.T_test)

            # record
            mu_list.append(mu)
            phi_list.append(phi)
            lamda_mu_list.append(lamda_mu)
            lamda_phi_list.append(lamda_phi)
            logl_train_list.append(logl_train)
            logl_test_list.append(logl_test)

        return mu_list,phi_list,lamda_mu_list,lamda_phi_list,logl_train_list,logl_test_list

#########################################################################################################

    'EM Algorithm'
    @staticmethod
    def gq_points_weights(a,b,Q):
        r"""
        Generate the Gaussian quadrature nodes and weights for the integral :math:`\int_a^b f(t) dt`

        :type a: float
        :param a: the lower end of the integral
        :type b: float
        :param b: the upper end of the integral
        :type Q: int
        :param Q: the number of Gaussian quadrature nodes (weights)
        :rtype: 1D numpy array, 1D numpy array
        :return: Gaussian quadrature nodes and the corresponding weights
        """
        p,w = legendre.leggauss(Q)
        c = np.array([0] * Q + [1])
        p_new = (a + b + (b - a) * p) / 2
        w_new = (b - a) / (legendre.legval(p, legendre.legder(c))**2*(1-p**2))
        return p_new,w_new

    def ini_P(self):
        r"""
        Initialize the probabilistic branching matrix. 

        :rtype: numpy array
        :return: probabilistic branching matrix, the flattened P_ij, interval of timestamps :\tau, the number of P_ij!=0 in each row 
        """
        N = len(self.points_hawkes)
        P = np.zeros((N,N))
        Pij_flat = []
        tau = []
        num_Pij_row = []
        for i in range(N):                    # initial value of P
            for j in range(i+1):
                tij = self.points_hawkes[i] - self.points_hawkes[j]
                if tij >= self.T_phi: continue
                else:
                    P[i][j:i+1] = np.random.dirichlet([1]*(i-j+1))
                    Pij_flat += list(P[i][j:i])
                    tau.append(list(self.points_hawkes[i] - np.array(self.points_hawkes[j:i])))
                    num_Pij_row.append(i-j)
                    break
        return P, Pij_flat, tau, num_Pij_row

    def a_predict(self, x_M, y_M, theta0, theta1, K_MM_inv, x_pred):
        r"""
        The mean of y_pred based on x_M, y_M (Gaussian process regression).
        
        :type x_M: numpy array
        :param x_M: input x
        :type y_M: numpy array
        :param y_M: input y
        :type theta0: 
        :param theta0:
        :type theta1: 
        :param theta1:
        :type K_MM_inv: 2D numpy array
        :param K_MM_inv: the inverse kernel matrix of x_M
        :type x_pred: numpy array
        :param x_pred: the predictive points

        :rtype: numpy array
        :return: mean of y_pred
        """
        k = self.rbf_kernel(theta0, theta1, 0, x_pred, x_M)
        k_C = np.dot(k, K_MM_inv)
        y_pred_mean = np.dot(k_C, y_M)
        return y_pred_mean

    def EM(self, num_gq_mu, num_gq_phi, num_pre_mu, num_pre_phi, num_iter):
        r"""
        EM algorithm which is used to estimate the MAP of lamda_ub_mu, lamda_ub_phi, g_s_mu and g_s_phi. 
        
        :type num_gq_mu: int
        :param num_gq_mu: the number of Gaussian quadrature nodes on [0,T]
        :type num_gq_phi: int
        :param num_gq_phi: the number of Gaussian quadrature nodes on [0,T_phi]
        :type num_pre_mu: int
        :param num_pre_mu: the number of prediction points on [0,T]
        :type num_pre_phi: int
        :param num_pre_phi: the number of prediction points on [0,T_phi]
        :type num_iter: int
        :param num_iter: the number of EM iterations

        :rtype: numpy array
        :return: the MAP estimates of \mu(t), \phi(\tau), lamda_ub_mu, lamda_ub_phi, the training and 
        test log-likelihood along EM iterations. 
        """
        N = len(self.points_hawkes)
        P, Pij_flat, tau_phi_md, num_Pij_row = self.ini_P()
        tau_phi = sum(tau_phi_md,[])
        N_phi = len(tau_phi)
        M_mu = len(self.ind_p_mu)
        M_phi = len(self.ind_p_phi)

        K_MM_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, self.noise_var_mu, self.ind_p_mu, self.ind_p_mu)
        K_MM_mu_inv = np.linalg.inv(K_MM_mu)
        K_MM_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi, self.noise_var_phi, self.ind_p_phi, self.ind_p_phi)
        K_MM_phi_inv = np.linalg.inv(K_MM_phi)

        K_NM_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, 0, np.array(self.points_hawkes), self.ind_p_mu)
        K_NM_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi, 0, np.array(tau_phi), self.ind_p_phi)

        # initial gm_mu and lamda_mu, gm_phi and lamda_phi
        gm_mu = np.random.uniform(-1, 1, size = M_mu)
        gm_phi = np.random.uniform(-1, 1, size = M_phi)
        lamda_mu = sum(np.diag(P))*2/self.T
        lamda_phi = sum(Pij_flat)*2/N/self.T_phi

        # gaussian quadreture points and weights
        p_gq_mu, w_gq_mu = self.gq_points_weights(0,self.T,num_gq_mu)
        p_gq_phi, w_gq_phi = self.gq_points_weights(0,self.T_phi,num_gq_phi)

        K_gqM_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, 0, p_gq_mu, self.ind_p_mu)
        K_gqM_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi, 0, p_gq_phi, self.ind_p_phi)

        mu_list=[]
        phi_list=[]
        lamda_mu_list=[]
        lamda_phi_list=[]
        logl_train_list=[]
        logl_test_list=[]

        for iteration in range(num_iter):
            # update distribution of w_ii and w_ij
            a_ii = self.a_predict(self.ind_p_mu, gm_mu, self.theta0_mu, self.theta1_mu, K_MM_mu_inv, np.array(self.points_hawkes))
            E_w_ii = 1/2/a_ii*np.tanh(a_ii/2)
            a_ij = self.a_predict(self.ind_p_phi, gm_phi, self.theta0_phi, self.theta1_phi, K_MM_phi_inv, np.array(tau_phi))
            E_w_ij = 1/2/a_ij*np.tanh(a_ij/2)
            
            # update lamda_mu and lamda_phi
            a_gq_mu = self.a_predict(self.ind_p_mu, gm_mu, self.theta0_mu, self.theta1_mu, K_MM_mu_inv, p_gq_mu)
            int_intensity = np.sum(w_gq_mu*lamda_mu*expit(-a_gq_mu))
            lamda_mu = (np.sum(np.diag(P))+int_intensity)/self.T
            
            a_gq_phi = self.a_predict(self.ind_p_phi, gm_phi, self.theta0_phi, self.theta1_phi, K_MM_phi_inv, p_gq_phi)
            int_intensity = np.sum(w_gq_phi*lamda_phi*expit(-a_gq_phi))
            lamda_phi = (np.sum(Pij_flat) + N*int_intensity)/N/self.T_phi
            
            # update gm_mu and gm_phi
            int_A_mu=np.zeros((M_mu,M_mu))
            for i in range(N):
                int_A_mu+=P[i][i]*E_w_ii[i]*np.outer(K_NM_mu[i],K_NM_mu[i])
            for i in range(num_gq_mu):
                int_A_mu+=w_gq_mu[i]*(lamda_mu/2/a_gq_mu[i]*np.tanh(a_gq_mu[i]/2)*expit(-a_gq_mu[i])*np.outer(K_gqM_mu[i],K_gqM_mu[i]))
            int_B_mu=np.zeros(M_mu)
            for i in range(N):
                int_B_mu+=0.5*P[i][i]*K_NM_mu[i]
            for i in range(num_gq_mu):
                int_B_mu+=-w_gq_mu[i]/2*(lamda_mu*expit(-a_gq_mu[i])*K_gqM_mu[i])
            gm_mu=np.dot(np.dot(np.linalg.inv(np.dot(np.dot(K_MM_mu_inv,int_A_mu),K_MM_mu_inv)+K_MM_mu_inv),K_MM_mu_inv),int_B_mu)
            
            int_A_phi=np.zeros((M_phi,M_phi))
            for i in range(N_phi):
                int_A_phi+=Pij_flat[i]*E_w_ij[i]*np.outer(K_NM_phi[i],K_NM_phi[i])
            for i in range(num_gq_phi):
                int_A_phi+=w_gq_phi[i]*N*lamda_phi/2/a_gq_phi[i]*np.tanh(a_gq_phi[i]/2)*expit(-a_gq_phi[i])*np.outer(K_gqM_phi[i],K_gqM_phi[i])
            int_B_phi=np.zeros(M_phi)
            for i in range(N_phi):
                int_B_phi+=0.5*Pij_flat[i]*K_NM_phi[i]
            for i in range(num_gq_phi):
                int_B_phi+=-w_gq_phi[i]/2*N*lamda_phi*expit(-a_gq_phi[i])*K_gqM_phi[i]
            gm_phi=np.dot(np.dot(np.linalg.inv(np.dot(np.dot(K_MM_phi_inv,int_A_phi),K_MM_phi_inv)+K_MM_phi_inv),K_MM_phi_inv),int_B_phi)
            
            # update \mu, \phi and P
            Pij_flat=[]
            for i in range(N): # updata of P
                mu_ti=lamda_mu*expit(a_ii[i])
                phi_ti=lamda_phi*expit(a_ij[sum(num_Pij_row[:i]):sum(num_Pij_row[:i+1])])
                intensity_total=mu_ti+np.sum(phi_ti)
                P[i][i]=mu_ti/intensity_total
                P_i_j=phi_ti/intensity_total
                P[i][i-len(phi_ti):i]=P_i_j
                Pij_flat+=list(P_i_j)

            # compute g_{\mu}(t) and g_{\phi}(\tau) on finer grid and the corresponding train/test loglikelihood
            g_mu_em = self.a_predict(self.ind_p_mu, gm_mu, self.theta0_mu, self.theta1_mu, K_MM_mu_inv, np.linspace(0, self.T, num_pre_mu))
            g_phi_em = self.a_predict(self.ind_p_phi, gm_phi, self.theta0_phi, self.theta1_phi, K_MM_phi_inv, np.linspace(0, self.T_phi, num_pre_phi))
            mu_em = lamda_mu*expit(g_mu_em)
            phi_em = lamda_phi*expit(g_phi_em)
            logl_train = self.loglikelihood_discrete_phi_mu(self.points_hawkes, mu_em, phi_em, self.T_phi, self.T)
            logl_test = self.loglikelihood_discrete_phi_mu(self.points_hawkes_test, mu_em, phi_em, self.T_phi, self.T_test)

            # record
            mu_list.append(mu_em)
            phi_list.append(phi_em)
            lamda_mu_list.append(lamda_mu)
            lamda_phi_list.append(lamda_phi)
            logl_train_list.append(logl_train)
            logl_test_list.append(logl_test)
        return mu_list,phi_list,lamda_mu_list,lamda_phi_list,logl_train_list,logl_test_list

#########################################################################################################

    'Mean-Field Variational Inference'

    def a_c_predict(self, x_M, y_M_mean, y_M_cov, theta0, theta1, noise_var, K_MM_inv, x_pred): 
        r"""
        The mean, variance and E[y_pred^2] of y_pred based on x_M, y_M (Gaussian process regression). 

        :type x_M: numpy array
        :param x_M: input x
        :type y_M_mean: numpy array
        :param y_M_mean: input y mean
        :type y_M_cov: 2D numpy array
        :param y_M_cov: input y covariance
        :type theta0: 
        :param theta0:
        :type theta1: 
        :param theta1:
        :type noise_var: 
        :param noise_var: 
        :type K_MM_inv: 2D numpy array
        :param K_MM_inv: the inverse kernel matrix of x_M
        :type x_pred: numpy array
        :param x_pred: the predictive points

        :rtype: numpy arrays
        :return: mean, variance and E[y_pred^2] of y_pred
        """
        k = self.rbf_kernel(theta0, theta1, noise_var, x_pred, x_M)
        k_C = np.dot(k, K_MM_inv)
        y_pred_mean = np.dot(k_C, y_M_mean)
        k_matrix_pre = self.rbf_kernel(theta0, theta1, noise_var, x_pred, x_pred)
        y_pred_cov = k_matrix_pre - np.dot(k_C,k.T) + np.dot(np.dot(k_C, y_M_cov), k_C.T)
        return y_pred_mean, np.sqrt(np.diag(y_pred_cov)+y_pred_mean**2), y_pred_cov

    @staticmethod
    def E_log_sigmoid(mean, variance, p, w):
        r"""
        Compute the \math: \int \log(\sigma(x))*N(x|mean, variance)dx

        :type mean: float
        :param mean: the mean of normal distribution
        :type variance: float 
        :param variance: the variance of normal distribution
        :type p: numpy array
        :param p: Gaussian quadrature nodes
        :type w: numpy array
        :param w: Gaussian quadrature weights

        :rtype: float
        :return: the integral value
        """
        return -np.sum(w*np.log(1+np.exp(-mean-np.sqrt(variance)*p))*np.exp(-p**2/2))/np.sqrt(2*np.pi)

    def MF(self, num_gq_mu, num_gq_phi, num_pre_mu, num_pre_phi, num_iter):
        r"""
        Mean-field variational inference algorithm which is used to estimate the posterior of lamda_ub_mu, lamda_ub_phi, g_s_mu and g_s_phi. 
        
        :type num_gq_mu: int
        :param num_gq_mu: the number of Gaussian quadrature nodes on [0,T]
        :type num_gq_phi: int
        :param num_gq_phi: the number of Gaussian quadrature nodes on [0,T_phi]
        :type num_pre_mu: int
        :param num_pre_mu: the number of prediction points on [0,T]
        :type num_pre_phi: int
        :param num_pre_phi: the number of prediction points on [0,T_phi]
        :type num_iter: int
        :param num_iter: the number of MF iterations

        :rtype: numpy array
        :return: the  of \mu(t), \phi(\tau), lamda_ub_mu, lamda_ub_phi, the training and 
        test log-likelihood along MF iterations. 
        """
        N = len(self.points_hawkes)
        P, Pij_flat, tau_phi_md,_ = self.ini_P()
        tau_phi = sum(tau_phi_md,[])
        N_phi = len(tau_phi)
        M_mu = len(self.ind_p_mu)
        M_phi = len(self.ind_p_phi)

        K_MM_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, self.noise_var_mu, self.ind_p_mu, self.ind_p_mu)
        K_MM_mu_inv = np.linalg.inv(K_MM_mu)
        
        K_MM_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi,self.noise_var_phi, self.ind_p_phi, self.ind_p_phi)
        K_MM_phi_inv = np.linalg.inv(K_MM_phi)
        
        K_NM_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, 0, np.array(self.points_hawkes), self.ind_p_mu)
        K_NM_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi, 0, np.array(tau_phi), self.ind_p_phi)
        
        # initial q_gm_mu q_gm_phi and q_lamda_mu q_lamda_phi
        # q_lamda_mu is a gamma distribution gamma(alpha=sum(P_ii)+E(|pi|),scale=1/T)
        alpha_mu = sum(np.diag(P))*2
        # q_lamda_phi is a gamma distribution gamma(alpha=sum(P_ij)+N*E(|pi_n|),scale=1/N/T_phi)
        alpha_phi = sum(Pij_flat)*2
        # q_gm_mu is a gaussian distribution N(mean_gm_mu,cov_gm_mu)
        mean_gm_mu = np.random.uniform(-1, 1, size = M_mu)
        cov_gm_mu = K_MM_mu
        mean_gm_phi = np.random.uniform(-1, 1, size = M_phi)
        cov_gm_phi = K_MM_phi
        
        # gaussian quadreture points and weights
        p_gq_mu, w_gq_mu = self.gq_points_weights(0, self.T, num_gq_mu)
        p_gq_phi, w_gq_phi = self.gq_points_weights(0, self.T_phi, num_gq_phi)
        
        E_log_sig_gm_mu=np.zeros(M_mu)
        E_log_sig_gm_phi=np.zeros(M_phi)
        p_gq_E_log_sig, w_gq_E_log_sig = self.gq_points_weights(-6,6,100) # gaussian quadrature nodes to compute the \int \log(\sigma(x))*N(x|mean, variance)dx
        
        K_gqM_mu = self.rbf_kernel(self.theta0_mu, self.theta1_mu, 0, p_gq_mu, self.ind_p_mu)
        K_gqM_phi = self.rbf_kernel(self.theta0_phi, self.theta1_phi, 0, p_gq_phi, self.ind_p_phi)
        
        logl_train_list=[]
        logl_test_list=[]
        
        for iteration in range(num_iter):
            # update parameters of density of q_wij: PG(wij|1,cij)
            _, c_ii, _ = self.a_c_predict(self.ind_p_mu, mean_gm_mu, cov_gm_mu, self.theta0_mu, self.theta1_mu, self.noise_var_mu, K_MM_mu_inv, np.array(self.points_hawkes))
            E_w_ii = 1/2/c_ii*np.tanh(c_ii/2)
            _, c_ij, _ = self.a_c_predict(self.ind_p_phi, mean_gm_phi, cov_gm_phi, self.theta0_phi, self.theta1_phi, self.noise_var_phi, K_MM_phi_inv, np.array(tau_phi))
            E_w_ij = 1/2/c_ij*np.tanh(c_ij/2)
            
            # update parameters of q_pi_mu intensity=exp(E(log lamda_mu))sigmoid(-c(t))exp((c(t)-a(t))/2)*P_pg(wii|1,c(t))
            # update parameters of q_pi_phi intensity=exp(E(log lamda_phi))sigmoid(-c(tau))exp((c(tau)-a(tau))/2)*P_pg(wij|1,c(tau))
            lamda_1_mu = np.exp(np.log(1/self.T)+psi(alpha_mu))
            lamda_1_phi = np.exp(np.log(1/N/self.T_phi)+psi(alpha_phi))
            
            # update parameters of q_lamda_mu q_lamda_mu=gamma(alpha=sum(pii)+E(|pi|),scale=1/T)
            # update parameters of q_lamda_phi q_lamda_phi=gamma(alpha=sum(pij)+N*E(|pi_phi|),scale=1/N/T_phi)
            a_gq_mu, c_gq_mu, _ = self.a_c_predict(self.ind_p_mu, mean_gm_mu, cov_gm_mu, self.theta0_mu, self.theta1_mu, self.noise_var_mu, K_MM_mu_inv, p_gq_mu)
            int_intensity = np.sum(w_gq_mu*lamda_1_mu*expit(-c_gq_mu)*np.exp((c_gq_mu-a_gq_mu)/2))
            alpha_mu = np.sum(np.diag(P)) + int_intensity
            
            a_gq_phi, c_gq_phi, _ = self.a_c_predict(self.ind_p_phi, mean_gm_phi, cov_gm_phi, self.theta0_phi, self.theta1_phi, self.noise_var_phi, K_MM_phi_inv, p_gq_phi)
            int_intensity = np.sum(w_gq_phi*lamda_1_phi*expit(-c_gq_phi)*np.exp((c_gq_phi-a_gq_phi)/2))
            alpha_phi = sum(Pij_flat) + N*int_intensity
            
            # update parameters of q_gm_mu  q_gm_mu=N(mean_gm_mu,cov_gm_mu)
            # update parameters of q_gm_phi q_gm_phi=N(mean_gm_phi,cov_gm_phi)
            int_A_mu=np.zeros((M_mu,M_mu))
            for i in range(N):
                int_A_mu+=P[i][i]*E_w_ii[i]*np.outer(K_NM_mu[i],K_NM_mu[i])
            for i in range(num_gq_mu):
                int_A_mu+=w_gq_mu[i]*(lamda_1_mu/2/c_gq_mu[i]*np.tanh(c_gq_mu[i]/2)*expit(-c_gq_mu[i])\
                                      *np.exp((c_gq_mu[i]-a_gq_mu[i])/2)*np.outer(K_gqM_mu[i],K_gqM_mu[i]))
            int_B_mu=np.zeros(M_mu)
            for i in range(N):
                int_B_mu+=0.5*P[i][i]*K_NM_mu[i]
            for i in range(num_gq_mu):
                int_B_mu+=-w_gq_mu[i]/2*(lamda_1_mu*expit(-c_gq_mu[i])*np.exp((c_gq_mu[i]-a_gq_mu[i])/2)*K_gqM_mu[i])
            cov_gm_mu = np.linalg.inv(np.dot(np.dot(K_MM_mu_inv,int_A_mu),K_MM_mu_inv)+K_MM_mu_inv)
            mean_gm_mu = np.dot(np.dot(cov_gm_mu,K_MM_mu_inv),int_B_mu)
            
            int_A_phi=np.zeros((M_phi,M_phi))
            for i in range(N_phi):
                int_A_phi+=Pij_flat[i]*E_w_ij[i]*np.outer(K_NM_phi[i],K_NM_phi[i])
            for i in range(num_gq_phi):
                int_A_phi+=w_gq_phi[i]*(N*lamda_1_phi/2/c_gq_phi[i]*np.tanh(c_gq_phi[i]/2)*expit(-c_gq_phi[i])\
                                        *np.exp((c_gq_phi[i]-a_gq_phi[i])/2)*np.outer(K_gqM_phi[i],K_gqM_phi[i]))
            int_B_phi=np.zeros(M_phi)
            for i in range(N_phi):
                int_B_phi+=0.5*Pij_flat[i]*K_NM_phi[i]
            for i in range(num_gq_phi):
                int_B_phi+=-w_gq_phi[i]/2*(N*lamda_1_phi*expit(-c_gq_phi[i])*np.exp((c_gq_phi[i]-a_gq_phi[i])/2)*K_gqM_phi[i])
            cov_gm_phi = np.linalg.inv(np.dot(np.dot(K_MM_phi_inv,int_A_phi),K_MM_phi_inv)+K_MM_phi_inv)
            mean_gm_phi = np.dot(np.dot(cov_gm_phi,K_MM_phi_inv),int_B_phi)
            
            # update mu, phi and P
            for i in range(M_mu):
                E_log_sig_gm_mu[i] = self.E_log_sigmoid(mean_gm_mu[i], cov_gm_mu[i][i], p_gq_E_log_sig, w_gq_E_log_sig)
            for i in range(M_phi):
                E_log_sig_gm_phi[i] = self.E_log_sigmoid(mean_gm_phi[i], cov_gm_phi[i][i], p_gq_E_log_sig, w_gq_E_log_sig)
            E_log_sig_g_mu = self.a_predict(self.ind_p_mu,E_log_sig_gm_mu,self.theta0_mu,self.theta1_mu,K_MM_mu_inv,np.linspace(0,self.T,num_pre_mu))
            E_log_sig_g_phi = self.a_predict(self.ind_p_phi,E_log_sig_gm_phi,self.theta0_phi,self.theta1_phi,K_MM_phi_inv,np.linspace(0,self.T_phi,num_pre_phi))
            
            mu = lamda_1_mu*np.exp(E_log_sig_g_mu)
            phi = lamda_1_phi*np.exp(E_log_sig_g_phi)
            
            Pij_flat=[]
            for i in range(N): # updata of P
                mu_ti=mu[int(self.points_hawkes[i]/(self.T/num_pre_mu))]
                phi_ti=[phi[index] for index in (np.array(tau_phi_md[i])/(self.T_phi/num_pre_phi)).astype(int)]
                intensity_total=sum(phi_ti)+mu_ti
                P[i][i]=mu_ti/intensity_total
                P_i_j=np.array(phi_ti)/intensity_total
                P[i][i-len(phi_ti):i]=P_i_j
                Pij_flat+=list(P_i_j)
            
            # train/test loglikelihood
            logl_train = self.loglikelihood_discrete_phi_mu(self.points_hawkes, mu, phi, self.T_phi, self.T)
            logl_test = self.loglikelihood_discrete_phi_mu(self.points_hawkes_test, mu, phi, self.T_phi, self.T_test)

            # record
            logl_train_list.append(logl_train)
            logl_test_list.append(logl_test)

        mean_g_mu,_,cov_g_mu = self.a_c_predict(self.ind_p_mu,mean_gm_mu,cov_gm_mu,self.theta0_mu,self.theta1_mu,self.noise_var_mu,K_MM_mu_inv,np.linspace(0,self.T,num_pre_mu))
        mean_g_phi,_,cov_g_phi = self.a_c_predict(self.ind_p_phi,mean_gm_phi,cov_gm_phi,self.theta0_phi,self.theta1_phi,self.noise_var_phi,K_MM_phi_inv,np.linspace(0,self.T_phi,num_pre_phi))
        lamda_mu_mean = alpha_mu/self.T
        lamda_mu_var = alpha_mu/(self.T**2)
        lamda_phi_mean = alpha_phi/(N*self.T_phi)
        lamda_phi_var = alpha_phi/((N*self.T_phi)**2)

        return mean_g_mu, cov_g_mu, mean_g_phi, cov_g_phi, lamda_mu_mean, lamda_mu_var, lamda_phi_mean, lamda_phi_var, logl_train_list, logl_test_list