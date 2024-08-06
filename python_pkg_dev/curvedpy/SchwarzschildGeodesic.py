
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time

class SchwarzschildGeodesic:
    

    def __init__(self):

        self.r_s_value = 1 # We keep the Schwarzschild radius at one and scale appropriately
        self.time_like = False # No Massive particle geodesics yet

        # Define symbolic variables
        self.t, self.x, self.y, self.z, self.r_s = sp.symbols('t x y z r_s')
        
        # Radial distance to BlackHole location
        self.R = sp.sqrt(self.x**2 + self.y**2 + self.z**2)
        
        # The Schwarzschild metric
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
                        k_x_0 = 1., k_y_0 = 0., k_z_0 = 0., \
                        x0 = -10.0, y0 = 5.0, z0 = 5.0, \
                        curve_start = 0, \
                        curve_end = 50, \
                        nr_points_curve = 50, \
                        verbose = True \
                       ):
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
            if verbose: print("Event Hit BH: ", x, y, z, self.r_s_value, x**2 + y**2 + z**2 - self.r_s_value**2)
            return x**2 + y**2 + z**2 - self.r_s_value**2
        hit_blackhole.terminal = True
        #hit_blackhole.direction = -1
        
        def hit_background(t, y):
            k_x, x, k_y, y, k_z, z = y
            return x-15. # !!!! DEZE WAARDE UITPROGRAMMEREN
        hit_background.terminal = False

        values_0 = [ k_x_0, x0, k_y_0, y0, k_z_0, z0 ]
        t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

        start = time.time()
        result = solve_ivp(step, (curve_start, curve_end), values_0, t_eval=t_pts, \
                           events=[hit_blackhole, hit_background])
        end = time.time()
        if verbose: print("New: ", result.message, end-start, "sec")
            
        result.update({"hit_background": len(result.t_events[1])>0})
        result.update({"hit_blackhole": len(result.t_events[0])>0})
        result.update({"hit_nothing": len(result.t_events[0]) == 0 and len(result.t_events[1]) == 0})

        return result


    # This function is used by Blender. It scales/adepts the coordinate systems for the calc_trajectory function
    # and scales and slices the results before giving it back to for example Blender
    def ray_trace(  self, direction, loc_hit, \
                    ratio_obj_to_blackhole = 20, \
                    exit_tolerance = 0.1, \
                    curve_end = -1, \
                    warnings = True, verbose=False):
        # loc_hit: the BH is assumed to be located at the origin and loc_hit is relative to the origin and thus location of the BH
        # R_obj_blender: the size of the object representing the black hole in Blender
        # exit_tolerance: the ray tracing stops when it exits the sphere of influence. You can change the size of the 
        # sphere a bit to determine where it exits. This is done using: x**2 + y**2 + z**2 < (R_influence*(exit_tolerance+1.0))**2


        direction = direction/np.linalg.norm(direction) # Normalize the direction of the photon
        loc_hit_original = loc_hit

        #loc_hit = np.array(loc_hit) # - np.array(loc_bh) # Get the hit coords relative to the BH
        #loc_hit_original_relBH = loc_hit

        R_sphere = np.linalg.norm(loc_hit) # The size of the sphere in Blender
        R_schwarz = R_sphere / ratio_obj_to_blackhole # The size of the BH in Blender

        scale_factor = 1/R_schwarz # Everything gets scaled by this factor to have Schwarzschild radius of 1

        loc_hit = loc_hit * scale_factor
        R_sphere_scaled = R_sphere * scale_factor
        
        # Here I scale curve_end because otherwise, with a large R_influence, 
        # the integrator does not reach the otherside of the sphere
        # nr_points_curve SCHALING DIT MOET BETER!
        if curve_end == -1:
            curve_end = int(50*R_sphere_scaled)#/10.)

        res = self.calc_trajectory(\
                        k_x_0 = direction[0], k_y_0 = direction[1], k_z_0 = direction[2], \
                        x0 = loc_hit[0], y0 = loc_hit[1], z0 = loc_hit[2], \
                        curve_end = curve_end, verbose=verbose) 


        k_x, x, k_y, y, k_z, z = res.y

        if verbose: print(res)
        if verbose: print("Start before cut: ", x[0], y[0], z[0])
        
        x = x/scale_factor #+ loc_bh[0]
        y = y/scale_factor #+ loc_bh[1]
        z = z/scale_factor #+ loc_bh[2]

        list_i = []
        for i in range(len(x)):
            if x[i]**2 + y[i]**2 + z[i]**2 < (R_sphere*(exit_tolerance+1.0))**2:
                list_i.append(i)

        list_i.append(list_i[-1]+1) # Add one more element outside

        if verbose: print("Start after cut: ", x[0], y[0], z[0])

        if len(list_i) == 0 or res["hit_blackhole"]:
            return x, y, z, [], []
        else:
            x = x[list_i]
            y = y[list_i]
            z = z[list_i]
            k_x = k_x[list_i]
            k_y = k_y[list_i]
            k_z = k_z[list_i]

            # Forced normalization on the end_dir since it gave errors in trig functions. But need to
            # see how much the direction from the integrator deviates from normalized.
            end_dir = np.array([k_x[-1], k_y[-1], k_z[-1]]) / np.linalg.norm(np.array([k_x[-1], k_y[-1], k_z[-1]]))
            end_loc = np.array([x[-1], y[-1], z[-1]])

            if verbose: print("Start after cut and rescaling: ", x[0], y[0], z[0])
            return x, y, z, end_loc, end_dir
