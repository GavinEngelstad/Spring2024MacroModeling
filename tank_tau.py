'''
Two Agent New Keysian Model code
'''

class TANK:
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
        s.mu = p.mu # percent of population thats hand to mouth
        s.B = p.B # amount of bonds

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
        s.BSss = s.B / (1-s.mu)
        s.piss = 1.
        s.Zss = 1.
        s.xiss = 1.
        s.Rss = 1/s.beta
        s.Iss = 1/s.beta
        s.Lambdass = (s.psi-1)/s.psi
        s.Wss = (s.psi-1)/s.psi

        # ## newtons method
        def F(X):
            Yss, CHss, CSss = X
            return s.jnp.array([
                1/s.phi**(1/s.chi)*((s.psi-1)/s.psi)**((s.chi+1)/s.chi) * (1-s.gbar - (s.beta-1)*s.B/(s.beta*Yss))**((s.chi+1)/s.chi) * CHss**(-s.eta/s.chi) + (1-s.gbar - (s.beta-1)*s.B/(s.beta*Yss))/s.psi *Yss - CHss,
                (1-s.gbar - (s.beta-1)*s.B/(s.beta*Yss))**(1/s.chi) * ((s.psi-1)/(s.phi*s.psi))**(1/s.chi) * (s.mu*CHss**(-s.eta/s.chi) + (1-s.mu)*CSss**(-s.eta/s.chi)) - Yss,
                (s.mu*CHss + (1-s.mu)*CSss) / (1-s.gbar) - Yss
            ])        
        s.Yss, s.CHss, s.CSss = s.ut.newton(F, s.jnp.ones(3))

        ## remaining ones
        s.Nss = s.Yss
        s.LHss = (1-s.gbar - (s.beta-1)*s.B/(s.beta*s.Yss))**(1/s.chi) * ((s.psi-1)/(s.phi*s.psi))**(1/s.chi) * s.CHss**(-s.eta/s.chi)
        s.LSss = (1-s.gbar - (s.beta-1)*s.B/(s.beta*s.Yss))**(1/s.chi) * ((s.psi-1)/(s.phi*s.psi))**(1/s.chi) * s.CSss**(-s.eta/s.chi)
        s.PAss = s.Yss / (1-s.theta*s.beta)
        s.PBss = s.Yss / (1-s.theta*s.beta)
        s.Gss = s.gbar*s.Yss
        s.tauss = (1-s.beta)*s.B/(s.beta*s.Yss) + s.gbar

        # combined
        s.XXss = s.jnp.array([s.Wss, s.Yss, s.CHss, s.CSss, s.Nss, s.LHss, s.LSss, s.BSss, s.Gss, s.tauss, s.Lambdass, s.piss, s.PAss, s.PBss, s.Iss, s.Rss, s.Zss, s.xiss, s.gss])
        s.epsilonss = s.jnp.zeros(3)

        # check that its the ss
        # assert s.jnp.isclose(s.F(s.XXss, s.XXss, s.XXss, s.epsilonss), 0, atol=1e-7).all()

        return s.XXss, s.epsilonss
    

    def get_SS(self):
        '''
        Returns the steady state to the equations.

        Order is: W, Y, CH, CS, N, LH, LS,  Lambda, pi, PA, PB, I, R, Z, xi
        '''
        return self.Xss, self.epsilonss
    
    
    def F(self, X_p, X, X_l, epsilon):
        '''
        Finds the values of all the functions at the given points.
        
        Function value is defines as LHS-RHS

        X_p: (Expected) Next period states
        X: This period states
        X_l: Last period states
        epsolon: this period error

        Order for states: W, Y, N, Lambda, pi, PA, PB, I, R, Z, xi
        Order for error: epsilon_Z, epsilon_xi
        '''
        s = self

        ## decompose inputs
        W_p, Y_p, CH_p, CS_p, N_p, LH_p, LS_p, BS_p, G_p, tau_p, Lambda_p, pi_p, PA_p, PB_p, I_p, R_p, Z_p, xi_p, g_p = X_p
        W, Y, CH, CS, N, LH, LS, BS, G, tau, Lambda, pi, PA, PB, I, R, Z, xi, g = X
        W_l, Y_l, CH_l, CS_l, N_l, LH_l, LS_l, BS_l, G_l, tau_l, Lambda_l, pi_l, PA_l, PB_l, I_l, R_l, Z_l, xi_l, g_l = X_l
        epsilon_Z, epsilon_xi, epsilon_g = epsilon

        ## calcuate functions (each corresponds to their row in the system in the appendix of the paper)
        eq1 = (1-tau)*(W*(LH - N) + Y) - CH
        eq2 = s.phi * LH**s.chi * CH**s.eta - (1-tau)*W
        eq3 = R_l * BS_l + (1-tau)*(W*(LS - N) + Y) - CS - BS
        eq4 = s.phi * LS**s.chi * CS**s.eta - (1-tau)*W
        eq5 = s.beta * R * CS**s.eta / CS_p**s.eta - 1
        eq6 = Z * Lambda - W
        eq7 = s.theta * pi**(s.psi-1) + (1-s.theta) * (PA/PB)**(1-s.psi) - 1
        eq8 = s.psi/(s.psi-1) * Lambda * Y + s.theta * 1/R * pi_p**s.psi * PA_p - PA
        eq9 = Y + s.theta * 1/R * pi_p**(s.psi-1) * PB_p - PB
        eq10 = Z * N - Y
        eq11 = tau*Y + (1-s.mu)*BS - G - (1-s.mu)*R_l*BS_l
        eq12 = g*Y - G
        eq13 = s.R_goal * pi**s.omega * xi - I
        eq14 = I / pi_p - R
        eq15 = s.mu*LH + (1-s.mu)*LS - N
        eq16 = s.mu*CH + (1-s.mu)*CS + G - Y
        eq17 = s.rho_Z*s.jnp.log(Z_l) + epsilon_Z - s.jnp.log(Z)
        eq18 = s.rho_xi*s.jnp.log(xi_l) + epsilon_xi - s.jnp.log(xi)
        eq19 = s.rho_g*g_l + (1-s.rho_g)*s.gbar + epsilon_g - g

        return s.jnp.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19])
    

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
    # 0: W, 1: Y, 2: CH, 3: CS, 4: N, 5: LH, 6: LS, 7: BS, 8: G, 9: tau, 10: Lambda, 11: pi, 12: PA, 13: PB, 14: I, 15: R, 16: Z, 17: xi, 18: g
    show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    # build rank and run impulse response
    tank = TANK(params)
    print(tank.XXss)
    irf = tank.run_impulse_response(30, 2)

    # plot it
    plt.plot(irf.T[:, show], label=np.array(['W', 'Y', 'CH', 'CS', 'N', 'LH', 'LS', 'BS', 'G', 'tau', 'Lambda', 'pi', 'PA', 'PB', 'I', 'R', 'Z', 'xi', 'g'])[show])
    plt.legend(loc='lower right')
    plt.show()
