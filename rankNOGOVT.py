'''
Representative Agent New Keysian Model code
'''

class RANK:
    import jax.numpy as jnp
    import numpy as np
    import utils as ut
    import jax

    def __init__(self, param_file):
        self.set_params(param_file)
        self.build_SS()
        self.build_matrix_equations()
        self.build_policy_function()


    def set_params(self, param_file):
        '''
        Define the parameters used within the rank based on the `param_file'.
        '''
        s = self # shorter syntax
        p = param_file

        ## Household
        s.eta = p.eta # elasticity of consumption (>0, neq 1)
        s.chi = p.chi # elastisity of leisure ()
        s.phi = p.phi # relative importance of leisure to consumption ()
        s.beta = p.beta # intertemporal discounting of utility (0-1)

        ## Firms 
        s.psi = p.psi # elasticity of substiution for int goods (geq 1)
        s.theta = p.theta # frac of firms that keep old price (0-1)

        ## Central Bank
        s.omega = p.omega # weight of inflation in setting interest rate
        s.R_goal = p.R_goal # goal interest rate

        ## Stochatic Variables
        s.rho_Z = p.rho_Z # regression to SS Z
        s.sigma_Z = p.sigma_Z # stdev in Z
        s.rho_xi = p.rho_xi # regression to SS xi
        s.sigma_xi = p.sigma_xi # stdev in xi
    

    def build_SS(self):
        '''
        Build and return the steady state values. Also, stores them in the class
        '''
        s = self

        ## calculate values (in order in appendix of paper)
        s.piss = 1.
        s.Zss = 1.
        s.xiss = 1.
        s.Rss = 1/s.beta
        s.Iss = 1/s.beta
        s.Lambdass = (s.psi-1)/s.psi
        s.Wss = (s.psi-1)/s.psi
        s.Yss = ((s.psi-1)/(s.psi*s.phi))**(1/(s.eta+s.chi))
        s.Nss = ((s.psi-1)/(s.psi*s.phi))**(1/(s.eta+s.chi))
        s.PAss = ((s.psi-1)/(s.psi*s.phi))**(1/(s.eta+s.chi))/(1-s.theta*s.beta)
        s.PBss = ((s.psi-1)/(s.psi*s.phi))**(1/(s.eta+s.chi))/(1-s.theta*s.beta)

        # combined
        s.XXss = s.np.array([s.Wss,s.Yss, s.Nss, s.Lambdass, s.piss, s.PAss, s.PBss, s.Iss, s.Rss, s.Zss, s.xiss])
        s.epsilonss = s.np.zeros(2)

        # check that its the ss
        assert s.np.isclose(s.F(s.XXss, s.XXss, s.XXss, s.epsilonss), 0).all()

        return s.XXss
    

    def get_SS(self):
        '''
        Returns the steady state to the equations.

        Order is: W, Y, N, Lambda, pi, PA, PB, I, R, Z, xi
        '''
        return self.Xss, self.epsilonss
    
    
    def F(self, X_p, X, X_l, epsilon):
        '''
        Finds the values of all the functions at the given points.
        
        Function value is defines as LHS-RHS

        X_p: (Expected) Next period states
        X: This period states
        X_l: Last period states
        Z_p: (Expected) Next period randomness
        Z: This period randomness

        Order for states: W, Y, N, Lambda, pi, PA, PB, I, R, Z, xi
        Order for error: epsilon_Z, epsilon_xi
        '''
        s = self

        ## decompose inputs
        W_p, Y_p, N_p, Lambda_p, pi_p, PA_p, PB_p, I_p, R_p, Z_p, xi_p = X_p
        W, Y, N, Lambda, pi, PA, PB, I, R, Z, xi = X
        W_l, Y_l, N_l, Lambda_l, pi_l, PA_l, PB_l, I_l, R_l, Z_l, xi_l = X_l
        epsilon_Z, epsilon_xi = epsilon

        ## calcuate functions (each corresponds to their row in the system in the appendix of the paper)
        eq1 = s.phi * N**s.chi * Y**s.eta - W
        eq2 = s.beta * R * Y**s.eta / Y_p**s.eta - 1
        eq3 = Z * W - Lambda
        eq4 = s.theta * pi**(s.psi-1) + (1-s.theta) * (PA/PB)**(1-s.psi) - 1
        eq5 = s.psi/(s.psi-1) * Lambda * Y + s.theta * 1/R * pi_p**s.psi * PA_p - PA
        eq6 = Y + s.theta * 1/R * pi_p**(s.psi-1) * PB_p - PB
        eq7 = Z * N - Y
        eq8 = s.R_goal * pi**s.omega * xi - I
        eq9 = I / pi_p - R
        eq10 = s.rho_Z*s.jnp.log(Z_l) + epsilon_Z - s.jnp.log(Z)
        eq11 = s.rho_xi*s.jnp.log(xi_l) + epsilon_xi - s.jnp.log(xi)

        return s.jnp.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11])
    

    def build_matrix_equations(self):
        '''
        Build the matricies in the matrix system.

        Linear time iteration from Rendahl (2017) 
        '''
        s = self

        ## build matricies
        s.AA = s.jax.jacobian(lambda x: s.F(x, s.XXss, s.XXss, s.epsilonss))(s.XXss)
        s.BB = s.jax.jacobian(lambda x: s.F(s.XXss, x, s.XXss, s.epsilonss))(s.XXss)
        s.CC = s.jax.jacobian(lambda x: s.F(s.XXss, s.XXss, x, s.epsilonss))(s.XXss)
        s.EE = s.jax.jacobian(lambda x: s.F(s.XXss, s.XXss, s.XXss, x))(s.epsilonss)

        return s.AA, s.BB, s.CC, s.EE
    

    def get_matrix_equations(self):
        '''
        Returns the matricies in the matrix system.

        Order is F, G, H, L, M, N
        '''
        return self.AA, self.BB, self.CC, self.EE


    def build_policy_function(self):
        '''
        Build the matricies in the policy function.

        Brute force apraoch from Uhlig (1999)
        '''
        s = self

        ## build it fr
        s.PP, s.QQ = s.ut.solve_system(s.AA, s.BB, s.CC, s.EE)

        return s.PP, s.QQ
    

    def get_policy_function(self):
        '''
        Returns the matricies in the policy function.

        Order is P, q
        '''
        return self.PP, self.QQ
    

    def run_impulse_response(self, T, i, mult = 1, pct=True, act=False):
        '''
        Runs an returns the output to an impulse response

        T is the number of periods the impulse response is for
        i is the variable thats shocked, 0 for Z, 1 for xi
        mult multiplies the sigma for the variable before running the impuse
        response
        pct says to convert it into percent or actual deviation from steady
        state
        act says to convert into into the actual values instead of deviations

        Returns the values over time as an array
        '''
        s = self

        sigma = [s.sigma_Z, s.sigma_xi][i]*mult
        res = s.ut.impulse_response(s.PP, s.QQ, T, i, sigma)
        if act:
            res = res + s.XXss[:, None]
        if pct:
            res = 100*res / s.XXss[:, None]

        return res
    

    def run_simulation(self, T, sigma=None, pct=True, act=False, seed=None):
        '''
        Runs an returns the output to a simulation

        T is the number of periods the impulse response is for
        sigma is an array containing the standard deviations of the shocks
        pct says to convert it into percent or actual deviation from steady
        state
        act says to convert into into the actual values instead of deviations

        Returns the values over time as an array
        '''
        s = self

        if sigma == None:
            sigma = s.np.array([s.sigma_Z, s.sigma_xi])
        res = s.ut.simulate(s.PP, s.QQ, T, sigma, seed=seed)
        if act:
            res = res + s.XXss[:, None]
        if pct:
            res = 100*res / s.XXss[:, None]

        return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import params

    # List of variables to plot
    # 0: W, 1: Y, 2: N, 3: Lambda, 4: pi, 5: PA, 6: PB, 7: I, 8: R, 9: Z, 10: xi
    show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # build rank and run impulse response
    rank = RANK(params)
    irf = rank.run_impulse_response(30, 1)

    # plot it
    plt.plot(irf.T[:, show], label=np.array(['W', 'Y', 'N', 'Lambda', 'pi', 'PA', 'PB', 'I', 'R', 'Z', 'xi'])[show])
    plt.legend(loc='lower right')
    plt.show()
