'''
Representative Agent New Keysian Model code
'''

class RANK:
    import jax.numpy as jnp
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
        s.B = p.B # amount of bonds offered

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
        s.gbar = p.gbar # ss percent of gdp on govt spending
        s.rho_g = p.rho_g # regression to ss g
        s.sigma_g = p.sigma_g # stdev in g
    

    def build_SS(self):
        '''
        Build and return the steady state values. Also, stores them in the class
        '''
        s = self

        ## calculate values (in order in appendix of paper)
        s.gss = s.gbar
        s.Bss = s.B
        s.piss = 1.
        s.Zss = 1.
        s.xiss = 1.
        s.Rss = 1/s.beta
        s.Iss = 1/s.beta
        s.Lambdass = (s.psi-1)/s.psi
        s.Wss = (s.psi-1)/s.psi

        ## newtons method
        if s.B == 0:
            s.Yss = ((s.psi-1)*(1-s.gbar)**(1-s.eta)/(s.psi*s.phi))**(1/(s.chi+s.eta))
        else:
            a = s.phi*(1-s.gbar)**s.eta
            b = (1-s.gbar)*(1-s.psi)/s.psi
            c = (s.psi-1)*(s.beta-1)*s.B/(s.psi*s.beta)
            s.Yss = s.ut.simp_newton(lambda Y: a*Y**(1+s.chi+s.eta) + b*Y + c, 1.)

        ## Rest of the system
        s.Nss = s.Yss
        s.PAss = s.Yss/(1-s.theta*s.beta)
        s.PBss = s.Yss/(1-s.theta*s.beta)
        s.Gss = s.gbar*s.Yss
        s.Css = (1-s.gbar)*s.Yss
        s.tauss = (1-s.beta)*s.B/(s.beta*s.Yss) + s.gbar

        # combined
        s.XXss = s.jnp.array([s.Wss, s.Yss, s.Css, s.Nss, s.Bss, s.Gss, s.tauss, s.Lambdass, s.piss, s.PAss, s.PBss, s.Iss, s.Rss, s.Zss, s.xiss, s.gss])
        s.epsilonss = s.jnp.zeros(3)

        # check that its the ss
        # assert s.jnp.isclose(s.F(s.XXss, s.XXss, s.XXss, s.epsilonss), 0, atol=1e-3).all()

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
        W_p, Y_p, C_p, N_p, B_p, G_p, tau_p, Lambda_p, pi_p, PA_p, PB_p, I_p, R_p, Z_p, xi_p, g_p = X_p
        W, Y, C, N, B, G, tau, Lambda, pi, PA, PB, I, R, Z, xi, g = X
        W_l, Y_l, G_l, N_l, B_l, G_l, tau_l, Lambda_l, pi_l, PA_l, PB_l, I_l, R_l, Z_l, xi_l, g_l = X_l
        epsilon_Z, epsilon_xi, epsilon_g = epsilon

        ## calcuate functions (each corresponds to their row in the system in the appendix of the paper)
        eq1 = R_l * B_l + (1-tau)*Y - C - B
        eq2 = s.phi * N**s.chi * C**s.eta - (1-tau)*W
        eq3 = s.beta * R * C**s.eta / C_p**s.eta - 1
        eq4 = Z * Lambda - W
        eq5 = s.theta * pi**(s.psi-1) + (1-s.theta) * (PA/PB)**(1-s.psi) - 1
        eq6 = s.psi/(s.psi-1) * Lambda * Y + s.theta * 1/R * pi_p**s.psi * PA_p - PA
        eq7 = Y + s.theta * 1/R * pi_p**(s.psi-1) * PB_p - PB
        eq8 = Z * N - Y
        eq9 = tau*Y + B - G - R_l*B_l
        eq10 = g*Y - G
        eq11 = s.R_goal * pi**s.omega * xi - I
        eq12 = I / pi_p - R
        eq13 = C + G - Y
        eq14 = s.rho_Z*s.jnp.log(Z_l) + epsilon_Z - s.jnp.log(Z)
        eq15 = s.rho_xi*s.jnp.log(xi_l) + epsilon_xi - s.jnp.log(xi)
        eq16 = s.rho_g*g_l + (1-s.rho_g)*s.gbar + epsilon_g - g

        return s.jnp.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16])
    

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

        sigma = [s.Zss*s.sigma_Z, s.xiss*s.sigma_xi, s.gss*s.sigma_g][i]*mult
        res = s.ut.impulse_response(s.PP, s.QQ, T, i, sigma)
        if act:
            res = res + s.XXss[:, None]
        if pct:
            res = 100*res / (s.XXss[:, None] + (s.XXss[:, None] == 0)) ## handle 0 ss

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
            sigma = s.jnp.array([s.Zss*s.sigma_Z, s.xiss*s.sigma_xi, s.gss*s.sigma_g])
        res = s.ut.simulate(s.PP, s.QQ, T, sigma, seed=seed)
        if act:
            res = res + s.XXss[:, None]
        if pct:
            res = 100*res / (s.XXss[:, None] + (s.XXss[:, None] == 0)) ## handle 0 ss

        return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import params

    # List of variables to plot
    # 0: W, 1: Y, 2: C, 3: N, 4: B, 5: G, 6: tau, 7: Lambda, 8: pi, 9: PA, 10: PB, 11: I, 12: R, 13: Z, 14: xi, 15: g
    show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]

    # build rank and run impulse response
    rank = RANK(params)
    print(rank.XXss)
    irf = rank.run_impulse_response(100, 0, pct=False)
    # irf = rank.run_simulation(50, pct=False, act=True)

    # plot it
    plt.plot(irf.T[:, show], label=np.array(['W', 'Y', 'C', 'N', 'B', 'G', 'tau', 'Lambda', 'pi', 'PA', 'PB', 'I', 'R', 'Z', 'xi', 'g'])[show])
    plt.legend(loc='lower right')
    plt.show()
