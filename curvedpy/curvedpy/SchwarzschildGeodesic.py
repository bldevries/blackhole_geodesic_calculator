
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time

class SchwarzschildGeodesic:
    def __init__(self, metric = "schwarzschild"):

        self.r_s_value = 1 # We keep the Schwarzschild radius at one and scale appropriately
        self.time_like = False # No Massive particle geodesics yet

        # Define symbolic variables
        self.t, self.x, self.y, self.z, self.r_s = sp.symbols('t x y z r_s')
        
        # Radial distance to BlackHole location
        self.R = sp.sqrt(self.x**2 + self.y**2 + self.z**2)

        self.n = sp.Matrix([\
            [-1, 0, 0, 0],\
            [0, 1, 0, 0],\
            [0, 0, 1, 0],\
            [0, 0, 0, 1]\
            ])

        # The Schwarzschild metric
        self.g_SW = sp.Matrix([\
            [-(1-self.r_s/(4*self.R))**2 / (1+self.r_s/(4*self.R))**2, 0, 0, 0],\
            [0, (1+self.r_s/(4*self.R))**4, 0, 0], \
            [0, 0, (1+self.r_s/(4*self.R))**4, 0], \
            [0, 0, 0, (1+self.r_s/(4*self.R))**4], \
              ])

        if metric == "flat":
            self.g = self.n
        elif metric == "schwarzschild":
            self.g = self.g_SW      
        
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
                        max_step = np.inf,\
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
                           events=[hit_blackhole, hit_background],\
                           max_step = max_step)
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
                    curve_end = 50, \
                    interpolation_meth = "new",\
                    max_step = np.inf,\
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
        R_sphere_scaled = R_sphere * scale_factor # WHat is this for?????
        
        # Here I scale curve_end because otherwise, with a large R_influence, 
        # the integrator does not reach the otherside of the sphere
        # nr_points_curve SCHALING DIT MOET BETER!
        if curve_end == -1:
            curve_end = int(50*R_sphere_scaled)#/10.)

        res = self.calc_trajectory(\
                        k_x_0 = direction[0], k_y_0 = direction[1], k_z_0 = direction[2], \
                        x0 = loc_hit[0], y0 = loc_hit[1], z0 = loc_hit[2], \
                        curve_end = curve_end, max_step = max_step, verbose=verbose) 

        k_x, x, k_y, y, k_z, z = res.y

        # Scale the trajectory back
        x = x/scale_factor #+ loc_bh[0]
        y = y/scale_factor #+ loc_bh[1]
        z = z/scale_factor #+ loc_bh[2]

        # If you hit a blackhole, just exit
        if res["hit_blackhole"]:
            return x, y, z, [], [], \
                {"results": res, \
                "hit_blackhole": res["hit_blackhole"], \
                "k": [k_x, k_y, k_z]}


        # Interpolation: we need to get the exit location and direction of the ray at R=R_sphere
        # I have not yet found a good interpolator that interpolates trajectories. So I have a bit
        # of a build around using interp from numpy. I find the data points around the exit and inter
        # polate in that range. There I can garanty that the data points are increasing/decreasing.
        # So first I find the indices around the exit point:
        i_in, i_out = 0,0
        for i in range(len(x)):
            if i != 0:
                if (x[i-1]**2 + y[i-1]**2 + z[i-1]**2 < (R_sphere*(exit_tolerance+1.0))**2) \
                    and \
                    (x[i]**2 + y[i]**2 + z[i]**2 >= (R_sphere*(exit_tolerance+1.0))**2):
                    i_in, i_out = i-1, i
        if i_in == 0 and i_out == 0:
            end_loc, end_dir = [], []
        else:
            # Then I take a range of indices over which to interpolate
            if i_in-5 < 0: i_in = 0
            else: i_in = i_in-5
            if i_out+5 > len(x): i_out = len(x)
            else: i_out = i_out + 5

            # Then I interpolate using the radius as variable
            R = np.sqrt(x**2 + y**2 + z**2)
            #print("piep", i_in, i_out, len(x), R_sphere, R[i_in:i_out], x[i_in:i_out])
            x_end = np.interp(R_sphere*(exit_tolerance+1.0), R[i_in:i_out], x[i_in:i_out])
            y_end = np.interp(R_sphere*(exit_tolerance+1.0), R[i_in:i_out], y[i_in:i_out])
            z_end = np.interp(R_sphere*(exit_tolerance+1.0), R[i_in:i_out], z[i_in:i_out])
            k_x_end = np.interp(R_sphere*(exit_tolerance+1.0), R[i_in:i_out], k_x[i_in:i_out])
            k_y_end = np.interp(R_sphere*(exit_tolerance+1.0), R[i_in:i_out], k_y[i_in:i_out])
            k_z_end = np.interp(R_sphere*(exit_tolerance+1.0), R[i_in:i_out], k_z[i_in:i_out])

            # This then gives me the end location and direction of the ray
            end_loc = np.array([x_end, y_end, z_end])
            end_dir = np.array([k_x_end, k_y_end, k_z_end])
            end_dir = end_dir / np.linalg.norm(end_dir)

            end_loc = self.rayExitError(loc_hit_original, end_loc)

        return x, y, z, end_loc, end_dir, \
                {"results": res, \
                "hit_blackhole": res["hit_blackhole"], \
                "k": [k_x, k_y, k_z]}
        # else:
        #     list_i = []
        #     for i in range(len(x)):
        #         if x[i]**2 + y[i]**2 + z[i]**2 < (R_sphere*(exit_tolerance+1.0))**2:
        #             list_i.append(i)

        #     list_i.append(list_i[-1]+1) # Add one more element outside
            
        #     end_dir = np.array([k_x[-1], k_y[-1], k_z[-1]]) / np.linalg.norm(np.array([k_x[-1], k_y[-1], k_z[-1]]))
        #     end_loc = np.array([x[-1], y[-1], z[-1]])

        #     if len(list_i) == 0 or res["hit_blackhole"]:
        #         end_loc, end_dir = [], []
        #     else:
        #         x = x[list_i]
        #         y = y[list_i]
        #         z = z[list_i]
        #         k_x = k_x[list_i]
        #         k_y = k_y[list_i]
        #         k_z = k_z[list_i]

        #         # Forced normalization on the end_dir since it gave errors in trig functions. But need to
        #         # see how much the direction from the integrator deviates from normalized.
        #         end_dir = np.array([k_x[-1], k_y[-1], k_z[-1]]) / np.linalg.norm(np.array([k_x[-1], k_y[-1], k_z[-1]]))
        #         end_loc = np.array([x[-1], y[-1], z[-1]])

        #         if verbose: print("Start after cut and rescaling: ", x[0], y[0], z[0])
        #         #return x, y, z, end_loc, end_dir, {"message": ""}




        #if verbose: print("Start after cut: ", x[0], y[0], z[0])

    def approximateCurveEnd(self, ratio_obj_to_blackhole):
        return 50 + 2*50*(ratio_obj_to_blackhole/20 -1)

    def rayExitError(self, loc, end_loc):
        correction_factor = 1.1
        if np.linalg.norm(loc) >= np.linalg.norm(end_loc):
            print("Warning (rayExitError): end_loc is not outside sphere. It has been corrected but should not happen. (hit_loc, end_loc) = ", (loc, end_loc))
            end_loc = end_loc/np.linalg.norm(end_loc) * np.linalg.norm(loc) * correction_factor
        return end_loc


    # Get the impact parameter and vector
    def getImpactParam(self, loc_hit, dir_hit):
        # We create a line extending the dir_hit vector
        line = list(zip(*[loc_hit + dir_hit*l for l in range(20)]))
        # This line is used to construct the impact_vector
        impact_vector = loc_hit - loc_hit.dot(dir_hit)*dir_hit
        # We save the length of the impact_vector. This is called the impact parameter in
        # scattering problems
        impact_par = np.linalg.norm(impact_vector)
        # We normalize the impact vector. This way we get, together with dir_hit, an 
        # othonormal basis
        if impact_par != 0:
            impact_vector_normed = impact_vector/impact_par # !!! Check this, gives errors sometimes
        else:
            impact_vector_normed = impact_vector

        return impact_vector_normed, impact_par


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ApproxSchwarzschildGeodesic:
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __init__(self, ratio_obj_to_blackhole = 20, exit_tolerance = 0.1, extra_factor_curve_end = 1):
        self.ratio_obj_to_blackhole = ratio_obj_to_blackhole
        self.exit_tolerance = exit_tolerance
        

        self.SW = SchwarzschildGeodesic()
        self.curve_end = extra_factor_curve_end*self.SW.approximateCurveEnd(self.ratio_obj_to_blackhole)

        print("Approximatation Class ApproxSchwarzschildGeodesic created")
        print("  - ratio_obj_to_blackhole: ", self.ratio_obj_to_blackhole)
        print("  - exit_tolerance: ", self.exit_tolerance)
        print("  - curve_end: ", self.curve_end)
        print("  - extra_factor_curve_end: ", extra_factor_curve_end)
        print(" Starting data creation for interpolations ... ")

        self.data = {str(ratio_obj_to_blackhole): self.makeDataForRayTracer()}
        print("   Done.")



    # Calculate the exit location and direction using the coordinates in the Impact Plane space
    def getOutput(self, end_loc_impact_basis, end_dir_impact_basis, dir_hit_normed, impact_vector_normed):
        end_loc = end_loc_impact_basis[0] * dir_hit_normed + end_loc_impact_basis[1] * impact_vector_normed
        end_dir = end_dir_impact_basis[0] * dir_hit_normed + end_dir_impact_basis[1] * impact_vector_normed
        
        return end_loc, end_dir
        

    def getCoordinatesImpactPlane(self, loc_hit, dir_hit, end_loc, end_dir):
        # We will create a basis from the impact vector and the hit direction
        # The hit direction (dir_hit): the vector giving the direction which with the sphere is hit
        # The impact vector: is the vector pointing from the origin and which is orthogonal 
        #   to the hit direction vector. It is created from the dir_hit vector and the origin:
        #   https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
        # We write the coordinates in this impact space (x, y) where x is the coordinate for the 
        # hit_dir basis vector and y that for the impact_vector basis vector
        # Then the coordinates of the hit_dir vector in this space is (1, 0), since it is a basis vector
        
        # Make sure it is normed
        dir_hit = dir_hit / np.linalg.norm(dir_hit) 
        
        # Get the normed impact vector and impact parameter
        impact_vector_normed, impact_par = self.SW.getImpactParam(loc_hit, dir_hit)
        
        end_dir_impact_basis = (dir_hit.dot(end_dir), impact_vector_normed.dot(end_dir))
        end_loc_impact_basis = (dir_hit.dot(end_loc), impact_vector_normed.dot(end_loc))
        
        return impact_par, end_dir_impact_basis, end_loc_impact_basis
        
    # This calculates a grid of exit locations and directions in terms of their coordinates in the
    # ImpactPlane. These are later interpolated for ray tracing
    def makeDataForRayTracer(self):
        R_out = self.ratio_obj_to_blackhole #20 #(R_schw = 1, ratio is 20)
        z0 = 0
        y = np.array(range(1+int(self.ratio_obj_to_blackhole)*100))/100

        list_b, list_end_dir_impact_basis, list_end_loc_impact_basis = [], [], []
        for y0 in y: 
            # A ray directed at the middle of the BH gets through.
            # This must be an integrator error I think and I need to test this
            # For now I add a small deviation for things to get out right.
            if y0 == 0.0: 
                y0 += 0.001

            # Calculate x0 based on y0 and z0
            x0 = -1* np.sqrt(R_out**2 - y0**2)

            loc_hit = np.array([x0, y0, z0])
            dir_hit = np.array([1, 0, 0])
            dir_hit = dir_hit/np.linalg.norm(dir_hit)

            x, y, z, end_loc, end_dir, mes = self.SW.ray_trace(dir_hit, loc_hit, ratio_obj_to_blackhole = self.ratio_obj_to_blackhole, curve_end=self.curve_end, exit_tolerance = self.exit_tolerance)

            if len(end_loc) != 0:
                end_loc = self.SW.rayExitError(loc_hit, end_loc)

            if len(end_loc) != 0:
                b, end_dir_impact_basis, end_loc_impact_basis = self.getCoordinatesImpactPlane(loc_hit, dir_hit, end_loc, end_dir)

                list_b.append(b)
                list_end_dir_impact_basis.append(end_dir_impact_basis)
                list_end_loc_impact_basis.append(end_loc_impact_basis)

        list_end_dir_impact_basis_x, list_end_dir_impact_basis_y = list(zip(*list_end_dir_impact_basis))
        list_end_loc_impact_basis_x, list_end_loc_impact_basis_y = list(zip(*list_end_loc_impact_basis))

        data = [list_b, list_end_dir_impact_basis_x, list_end_dir_impact_basis_y, \
                list_end_loc_impact_basis_x, list_end_loc_impact_basis_y]
        
        #print("Done")
        return data


    def generatedRayTracer(self, loc_hit, dir_hit):
        # Scaling
        R_sphere = np.linalg.norm(loc_hit)
        scale_factor = self.ratio_obj_to_blackhole/R_sphere
        loc_hit_original = loc_hit
        loc_hit = loc_hit * scale_factor

        # Getting the pre-calculated data
        list_b, \
        list_end_dir_impact_basis_x, \
        list_end_dir_impact_basis_y, \
        list_end_loc_impact_basis_x, \
        list_end_loc_impact_basis_y = self.data[str(self.ratio_obj_to_blackhole)]

        # Get the impact parameter and vector        
        impact_vector_normed, impact_par = self.SW.getImpactParam(loc_hit, dir_hit)

        mes = {"impact_par": impact_par, "impact_vector_normed": impact_vector_normed, "b": list_b}
        
        # If the impact parameter is smaller than all pre-calculated ones, we assume it hits
        # the blackhole
        mes.update({"hit_blackhole": False})
        if impact_par < list_b[0]:
            # Hit the BH
            mes["hit_blackhole"] = True
            return [], [], mes
        # If the impact parameter is larger than the pre-calculated values, you have inputted a
        # hit_loc outside the sphere
        elif impact_par > list_b[-1]:
            mes.update({"error": "Outside"})
            return [], [], mes
        else:
            # Interpolate the coordinates in the ImpactPlane
            end_dir_impact_basis = np.array([\
                                        np.interp(impact_par, xp=list_b, fp=list_end_dir_impact_basis_x), \
                                        np.interp(impact_par, xp=list_b, fp=list_end_dir_impact_basis_y)])

            end_loc_impact_basis = np.array([\
                                        np.interp(impact_par, xp=list_b, fp=list_end_loc_impact_basis_x), \
                                        np.interp(impact_par, xp=list_b, fp=list_end_loc_impact_basis_y)])
            
            # Translate the coordinates to exit location and direction
            end_loc_gen, end_dir_gen = self.getOutput(end_loc_impact_basis, end_dir_impact_basis, dir_hit, impact_vector_normed)
            
            # Scaling back
            #print(np.linalg.norm(end_loc_gen), scale_factor)
            end_loc_gen, end_dir_gen = end_loc_gen/scale_factor, end_dir_gen#/scale_factor

            end_loc_gen = self.SW.rayExitError(loc_hit_original, end_loc_gen)

            #print(np.linalg.norm(end_loc_gen))
            mes.update({"message": ""})
            return end_loc_gen, end_dir_gen, mes

