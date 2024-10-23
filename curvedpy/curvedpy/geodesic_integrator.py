
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing as mp


class GeodesicIntegrator:
    def __init__(self, metric = "schwarzschild", mass = 1.):

        # Schwarzschild radius together with mass in geometrized units
        self.M = mass
        self.r_s_value = 2*self.M 

        # Type of geodesic
        self.time_like = False # No Massive particle geodesics yet

        # Define symbolic variables
        self.t, self.x, self.y, self.z, self.r_s = sp.symbols('t x y z r_s')
        
        # Radial distance to BlackHole location
        self.R = sp.sqrt(self.x**2 + self.y**2 + self.z**2)

        # The implemented metrics:
        if metric == "flat":
            self.g = sp.Matrix([\
                [-1, 0, 0, 0],\
                [0, 1, 0, 0],\
                [0, 0, 1, 0],\
                [0, 0, 0, 1]\
                ])
        elif metric == "schwarzschild":
            self.g = sp.Matrix([\
                [-(1-self.r_s/(4*self.R))**2 / (1+self.r_s/(4*self.R))**2, 0, 0, 0],\
                [0, (1+self.r_s/(4*self.R))**4, 0, 0], \
                [0, 0, (1+self.r_s/(4*self.R))**4, 0], \
                [0, 0, 0, (1+self.r_s/(4*self.R))**4], \
              ])     
        
        # Connection Symbols
        self.gam_t = sp.Matrix([[self.gamma_func(0,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_x = sp.Matrix([[self.gamma_func(1,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_y = sp.Matrix([[self.gamma_func(2,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_z = sp.Matrix([[self.gamma_func(3,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        
        # Building up the geodesic equation: 
        # Derivatives: k_beta = d x^beta / d lambda
        self.k_t, self.k_x, self.k_y, self.k_z = sp.symbols('k_t k_x k_y k_z', real=True)
        self.k = [self.k_t, self.k_x, self.k_y, self.k_z]
    
        # Second derivatives: d k_beta = d^2 x^beta / d lambda^2
        self.dk_t = sum([- self.gam_t[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_x = sum([- self.gam_x[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_y = sum([- self.gam_y[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_z = sum([- self.gam_z[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])

        # Norm of k
        # the norm of k determines if you have a massive particle (-1), a mass-less photon (0) 
        # or a space-like curve (1)
        self.norm_k = self.g[0, 0]*self.k_t**2 + self.g[1,1]*self.k_x**2 + \
                        self.g[2,2]*self.k_y**2 + self.g[3,3]*self.k_z**2
        
        # Now we calculate k_t using the norm. This eliminates one of the differential equations.
        # time_like = True: calculates a geodesic for a massive particle (not implemented yet)
        # time_like = False: calculates a geodesic for a photon
        if (self.time_like):
            self.k_t_from_norm = sp.solve(self.norm_k+1, self.k_t)[1]
        else:
            self.k_t_from_norm = sp.solve(self.norm_k, self.k_t)[1]

        # Lambdify versions
        self.dk_x_lamb = sp.lambdify([self.k_x, self.x, self.k_y, self.y, self.k_z, self.z, \
                                      self.k_t, self.t, self.r_s], \
                                     self.dk_x, "numpy")
        self.dk_y_lamb = sp.lambdify([self.k_x, self.x, self.k_y, self.y, self.k_z, self.z, \
                                      self.k_t, self.t, self.r_s], \
                                     self.dk_y, "numpy")
        self.dk_z_lamb = sp.lambdify([self.k_x, self.x, self.k_y, self.y, self.k_z, self.z, \
                                      self.k_t, self.t, self.r_s], \
                                     self.dk_z, "numpy")
        self.k_t_from_norm_lamb = sp.lambdify([self.k_x, self.x, self.k_y, self.y, self.k_z, self.z, \
                                               self.r_s], self.k_t_from_norm, "numpy")
     
    # Connection Symbols
    def gamma_func(self, sigma, mu, nu):
        coord_symbols = [self.t, self.x, self.y, self.z]
        g_sigma_mu_nu = 0
        for rho in [0,1,2,3]:
            if self.g[sigma, rho] != 0:
                g_sigma_mu_nu += 1/2 * 1/self.g[sigma, rho] * (\
                                self.g[nu, rho].diff(coord_symbols[mu]) + \
                                self.g[rho, mu].diff(coord_symbols[nu]) - \
                                self.g[mu, nu].diff(coord_symbols[rho]) )
            else:
                g_sigma_mu_nu += 0
        return g_sigma_mu_nu


    # This function does the numerical integration of the geodesic equation using scipy's solve_ivp
    def calc_trajectory(self, \
                        k_x_0 = 1., x0 = -10.0, k_y_0 = 0., y0 = 5.0, k_z_0 = 0., z0 = 5.0,\
                        R_end = -1,\
                        curve_start = 0, \
                        curve_end = 50, \
                        nr_points_curve = 50, \
                        max_step = np.inf,\
                        verbose = False \
                       ):

        # IMPLEMENT: check if inside horizon!
        r0 = np.linalg.norm(np.array([x0, y0, z0]))
        if r0 > self.r_s_value:
            # Step function needed for solve_ivp
            def step(lamb, new):
                new_k_x, new_x, new_k_y, new_y, new_k_z, new_z = new

                new_k_t = self.k_t_from_norm_lamb(*new, self.r_s_value)
                new_dk_x = self.dk_x_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dx = new_k_x
                new_dk_y = self.dk_y_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dy = new_k_y
                new_dk_z = self.dk_z_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dz = new_k_z

                return( new_dk_x, dx, new_dk_y, dy, new_dk_z, dz)

            def hit_blackhole(t, y): 
                k_x, x, k_y, y, k_z, z = y
                if verbose: print("Test Event Hit BH: ", x, y, z, self.r_s_value, x**2 + y**2 + z**2 - self.r_s_value**2)
                return x**2 + y**2 + z**2 - self.r_s_value**2
            hit_blackhole.terminal = True

            def reached_end(t, y): 
                k_x, x, k_y, y, k_z, z = y
                if verbose: print("Test Event End: ", np.sqrt(x**2 + y**2 + z**2), R_end, x**2 + y**2 + z**2 - R_end**2)
                return x**2 + y**2 + z**2 - R_end**2
            reached_end.terminal = True
            
            values_0 = [ k_x_0, x0, k_y_0, y0, k_z_0, z0 ]
            if nr_points_curve == 0:
                t_pts = None
            else:
                t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

            start = time.time()
            events = [hit_blackhole]
            if R_end > r0 : events.append(reached_end)
            result = solve_ivp(step, (curve_start, curve_end), values_0, t_eval=t_pts, \
                               events=events,\
                               max_step = max_step)
            end = time.time()
            if verbose: print("New: ", result.message, end-start, "sec")


            result.update({"hit_blackhole": len(result.t_events[0])>0})
            result.update({"start_inside_hole": False})

        else:
            if verbose: print("Starting location inside the blackhole.")
            result = {"start_inside_hole": True}

        return result

