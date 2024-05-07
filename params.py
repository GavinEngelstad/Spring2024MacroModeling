'''
Parameters used accross the models
'''
## Household
eta = 1. # elasticity of consumption (>0, neq 1)
chi = 2. # elastisity of leisure 
phi = 2. # relative importance of leisure to consumption
beta = 0.995 # intertemporal discounting of utility (0-1)
mu = 0.5 # percentage of population thats hand-to-mouth

## Firms 
psi = 6. # elasticity of substiution for int goods (geq 1)
theta = 0.75 # frac of firms that keep old price (0-1)

## Government
B = 0. # total bonds offered

## Central Bank
omega = 1.5 # weight of inflation in setting interest rate
R_goal = 1/beta # goal interest rate
# ss to euler equation means I_ss = 1/beta and ss to taylor rule means I_ss = R_goal

## Stochatic Variables
rho_Z = 0.95 # regression to SS Z
sigma_Z = 0.01 # stdev in Z

rho_xi = 0.8 # regression to SS xi
sigma_xi = 0.01 # stdev in xi

gbar = 0.25 # ss percent of gdp on govt spending
rho_g = 0.9 # regression to ss g
sigma_g = 0.01 # stdev in g


