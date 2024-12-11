

from importlib import reload
import bpy
import mathutils
import numpy as np
import curvedpy
import os
import time
import random
import multiprocessing as mp
reload(curvedpy)

bl_info = {
    "name": "Relativistic Render Engine",
    "bl_label": "Relativistic Render Engine",
    #"blender": (4, 1, 0),
    "category": "Render",
}

class RelativisticRenderEngine(bpy.types.RenderEngine):
    # Inspired by: https://docs.blender.org/api/current/bpy.types.RenderEngine.html?highlight=renderengine
    bl_idname = "RelRenEn"
    bl_label = "Relativistic"
    bl_use_preview = True
    
    textures = {"background": {"file_name": "8k_stars_milky_way.jpg", "texture_name": "8k_stars_milky_way.jpg"},\
                "background_ALT": {"file_name": "8k_stars_milky_way_ALT.png", "texture_name": "8k_stars_milky_way_ALT.png"},\
                "bg_ngc3293": {"file_name": "high_ngc3293_eso_8682×8436.jpg", "texture_name": "high_ngc3293_eso_8682×8436.jpg"},\
                "puppis": {"file_name": "ThreeClustersPuppis1824×1447.jpg", "texture_name": "ThreeClustersPuppis1824×1447.jpg"},\
                "high_PIA23647": {"file_name": "high_PIA23647.png", "texture_name": "high_PIA23647.png"},\
                "perseus-cluster": {"file_name": "high_1-Perseus-cluster_1oEasJg_6500×6500.jpg", "texture_name": "high_1-Perseus-cluster_1oEasJg_6500×6500.jpg"},\
                "moon": {"file_name": "8k_moon.jpg", "texture_name": "8k_moon.jpg"},\
                "test": {"file_name": "test.png", "texture_name": "test"},\
                "disk_clouds": {"file_name": "clouds_seamless_1024-512.png", "texture_name": "clouds_seamless_1024-512"},\
                "disk_clouds_high_contr": {"file_name": "clouds_seamless_2024_512_high_contr.png", "texture_name": "clouds_seamless_2024_512_high_contr"},\
                "disk_clouds_high_contr_color1": {"file_name": "clouds_seamless_2024_512_high_contr_color1.png", "texture_name": "clouds_seamless_2024_512_high_contr_color1"},\
                "HDR_blue_nebulae-1": {"file_name": "HDR_blue_nebulae-1.hdr", "texture_name": "HDR_blue_nebulae-1.hdr"},\
                "kloppenheim_02_puresky_8k.hdr": {"file_name": "kloppenheim_02_puresky_8k.hdr", "texture_name": "kloppenheim_02_puresky_8k.hdr"},\
                "TychoSkymap_t4_04096x02048.jpg":  {"file_name": "TychoSkymap_t4_04096x02048.jpg", "texture_name": "TychoSkymap_t4_04096x02048.jpg"},\
                "free_star_sky_hdri_spherical_map_by_kirriaa_dbw8p0w.jpg":  {"file_name": "free_star_sky_hdri_spherical_map_by_kirriaa_dbw8p0w.jpg", "texture_name": "free_star_sky_hdri_spherical_map_by_kirriaa_dbw8p0w.jpg"},\
                "annotated_free_star_sky.png":  {"file_name": "annotated_free_star_sky.png", "texture_name": "annotated_free_star_sky.png"},\
                "AdobeStock_417021435_sharpclouds_highdynamic_12k.jpeg":  {"file_name": "AdobeStock_417021435_sharpclouds_highdynamic_12k.jpeg", "texture_name": "AdobeStock_417021435_sharpclouds_highdynamic_12k.jpeg"},\
                "AdobeStock_890147410_star_clusters_purple_12k.jpeg":  {"file_name": "AdobeStock_890147410_star_clusters_purple_12k.jpeg", "texture_name": "AdobeStock_890147410_star_clusters_purple_12k.jpeg"},\
                "AdobeStock_1025944431_clouds_blue_dynamic_medium_12k.jpeg":  {"file_name": "AdobeStock_1025944431_clouds_blue_dynamic_medium_12k.jpeg", "texture_name": "AdobeStock_1025944431_clouds_blue_dynamic_medium_12k.jpeg"},\
                "test_areas.png":  {"file_name": "test_areas.png", "texture_name": "test_areas.png"},\
                "AdobeStock_890147410_star_clusters_purple_12k_MOON.png":  {"file_name": "AdobeStock_890147410_star_clusters_purple_12k_MOON.png", "texture_name": "AdobeStock_890147410_star_clusters_purple_12k_MOON.png"},\
                # "":  {"file_name": "", "texture_name": ""},\
                }

    # TychoSkymap_t4_04096x02048: https://svs.gsfc.nasa.gov/3442
    # RandomizedSkymap.t5_16384x08192: https://svs.gsfc.nasa.gov/4451/
    # free_star_sky_hdri_spherical_map_by_kirriaa_dbw8p0w https://www.deviantart.com/kirriaa/art/Free-star-sky-HDRI-spherical-map-719281328
    
    texture_dir ='/Users/vries001/Dropbox/0_DATA_BEN/PHYSICS/PROJECTS/blackhole_geodesic_calculator/raytracer/textures/'

    # ############################################################################################################################
    def render(self, depsgraph):
    # ############################################################################################################################

        print("Starting Render")
        #self.ratio_obj_to_blackhole = depsgraph.scene.ratio_obj_to_blackhole
        #self.exit_tolerance = depsgraph.scene.exit_tolerance

        self.max_integration_step = depsgraph.scene.max_integration_step
        self.sampling_seed = depsgraph.scene.sampling_seed

        # self.disk_on = depsgraph.scene.disk_on
        # self.disk_R_in = depsgraph.scene.disk_R_in
        # self.disk_R_out = depsgraph.scene.disk_R_out
        # self.disk_phase = depsgraph.scene.disk_phase
        # self.disk_mean = depsgraph.scene.disk_mean
        # self.disk_stddev = depsgraph.scene.disk_stddev
        # self.disk_intensity = depsgraph.scene.disk_intensity
        
        #self.approx = depsgraph.scene.approx
        self.back_ground_texture_key = depsgraph.scene.back_ground_texture_key
        #self.disk_texture = depsgraph.scene.disk_texture
        self.metric = depsgraph.scene.metric
        self.mass = depsgraph.scene.mass #0.5 # Thus R_horizon = 2*M in Geometrized units

        if self.max_integration_step == -1:
            self.max_integration_step = np.inf 

        self.scale = depsgraph.scene.render.resolution_percentage / 100.0
        self.res_x = int(depsgraph.scene.render.resolution_x*self.scale)
        self.res_y = int(depsgraph.scene.render.resolution_y*self.scale)
        
        self.field_of_view_x = depsgraph.scene.field_of_view_x
        self.field_of_view_y = depsgraph.scene.field_of_view_y
        
        self.samples = bpy.data.scenes['Scene'].eevee.taa_render_samples
        
        #self.blackhole_obj = depsgraph.scene.blackhole_obj
        if depsgraph.scene.blackhole_obj == None:
            self.bh_loc = mathutils.Vector([0, 0, 0])
        else:
            self.bh_loc = depsgraph.scene.blackhole_obj.location

        self.mark_y_min = depsgraph.scene.mark_y_min
        self.mark_y_max = depsgraph.scene.mark_y_max
        self.mark_x_min = depsgraph.scene.mark_x_min
        self.mark_x_max = depsgraph.scene.mark_x_max

        if self.mark_y_min == -1: self.mark_y_min = 0
        else: print("  - mark_y_min: ", self.mark_y_min)
        if self.mark_y_max == -1: self.mark_y_max = self.res_y
        else: print("  - mark_y_max: ", self.mark_y_max)
        if self.mark_x_min == -1: self.mark_x_min = 0
        else: print("  - mark_x_min: ", self.mark_x_min)
        if self.mark_x_max == -1: self.mark_x_max = self.res_x
        else: print("  - mark_x_max: ", self.mark_x_max)

        # Keep absolute pixel numbers, so this is commented out
        # self.mark_y_min = self.mark_y_min * res_y
        # self.mark_y_max = self.mark_y_max * res_y
        # self.mark_x_min = self.mark_x_min * res_x
        # self.mark_x_max = self.mark_x_max * res_x

        #self.debug_string = ""

        #print("  - Ratio: ", self.ratio_obj_to_blackhole)
        #print("  - Exit tol.: ", self.exit_tolerance)
        print("  - Integr. step: ", self.max_integration_step)
        #print("  - Approx. on: : ", self.approx)
        print("  - Metric: ", self.metric)
        print("  - Mass: ", self.mass)
        print("  - BH location: ", self.bh_loc)
        print("  - Sampling seed: ", self.sampling_seed)
        print("  - Samples: ", self.samples)
        print("  - BG texture: ", self.back_ground_texture_key)


        # if self.disk_on:
        #     print("  - Disk on: ", self.disk_on)
        #     print("  - Disk Rin: ", self.disk_R_in)
        #     print("  - Disk Rout: ", self.disk_R_out)
        #     print("  - Disk phase: ", self.disk_phase)
        #     print("  - Disk disk_mean: ", self.disk_mean)
        #     print("  - Disk disk_stddev: ", self.disk_stddev)
        #     print("  - Disk disk_intensity: ", self.disk_intensity)
        #     print("  - Disk texture: ", self.disk_texture)

        # Initiate the geodesic solver
        #self.SW = curvedpy.SchwarzschildGeodesic(metric=self.metric)
        #self.GeoInt = curvedpy.GeodesicIntegrator(metric = self.metric, mass = self.mass)
        self.GeoInt = curvedpy.GeodesicIntegratorSchwarzschild(mass = self.mass, time_like = False, verbose=False)


        # if self.disk_on and self.approx:
        #     print("WARNING: disk and approx are both on but do not work together. The disk is turned of.")
        #     self.disk_on = False

        # CHECK THIS!
        # if self.aSW != None:
        #     if round(self.exit_tolerance,4) != round(self.aSW.exit_tolerance, 4) or round(self.ratio_obj_to_blackhole, 4) != round(self.aSW.ratio_obj_to_blackhole, 4):
        #         print("Reloading approx data because settings have changed")
        #         self.aSW = curvedpy.ApproxSchwarzschildGeodesic(ratio_obj_to_blackhole = self.ratio_obj_to_blackhole, \
        #                                                            exit_tolerance = self.exit_tolerance)

        self.loadTextures()

        # For some reason I need to update the depsgraph otherwise things wont render 
        # from the console properly using "-f <frame_nr>"
        # This might be a horrible hack :S
        depsgraph = bpy.context.evaluated_depsgraph_get() 
        depsgraph.update()

        if self.is_preview:  # we might differentiate later
            pass             # for now ignore completely
        else:
            self.render_scene(depsgraph)#self.tex_name_bg, self.tex_name_stars)

    # ############################################################################################################################
    def render_scene(self, depsgraph):
    # ############################################################################################################################

        #width, height = self.res_x, self.res_y
        buf = np.ones(self.res_x*self.res_y*4)
        buf.shape = self.res_y,self.res_x,4

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.res_x, self.res_y)#self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]

        for y in self.ray_trace(depsgraph, self.res_x, self.res_y, 1, buf, self.samples): 
            buf.shape = -1,4  
            layer.rect = buf.tolist()  
            self.update_result(result)  
            buf.shape = self.res_y, self.res_x,4  
            self.update_progress(y/self.res_y)  
          
        self.end_result(result) 


    # ############################################################################################################################
    def ray_trace(self, depsgraph, width, height, depths, buf, samples, approx = False):
    # ############################################################################################################################
        # We collect the light objects in our scene            
        lamps = [ob for ob in depsgraph.scene.objects if ob.type == 'LIGHT']  

        # Buffer for the sample?
        sbuf = np.zeros(width*height*4)  
        sbuf.shape = height,width,4
        
        # This is the location and rotation of the camera (REFACTOR: CAM CAN BE PARENTED)
        origin = depsgraph.scene.camera.matrix_world.translation #.location  
        rotation = depsgraph.scene.camera.matrix_world.to_euler() #rotation_euler  
        
        aspectratio = height/width
        # Small deviations for the multisampling
        dy = aspectratio/height  
        dx = 1/width  
        random.seed(self.sampling_seed)  

        N = samples*width*height
        print("SAMPLING: samples:", samples, ", #rays total: ", N)

        start = time.time()
        for s in range(samples):
            print("  Sample #", s)
            sample_start = time.time()
            for y in range(height):
                if y >= self.mark_y_min and y <= self.mark_y_max:
                    y_render = (y-int(height/2))/height * aspectratio 

                    # Print some progress and start some timers
                    if y%50 == 0:
                        print("    Progress y: ", y, height)
                        row_start = time.time()
                        blender_functions_times = []
                        ray_trace_times = []
                        blackhole_hit_times = []

                    # pool = mp.Pool(mp.cpu_count())
                    # results = [pool.apply(self.spacetime_get_pixel, args=(x, y, dx, dy, width, height, depsgraph, origin, rotation)) for x in range(width)]
                    # print(results)
                    # pool.close()

                    # for i, x in enumerate(results):
                    #     sbuf[y,x,0:3] += results[i]

                    for x in range(width):
                        if x >= self.mark_x_min and x <= self.mark_x_max:

                            # get the direction.  
                            # camera points in -x direction, FOV = 90 degrees  
                            aspectratio = height/width
                            x_render = self.field_of_view_x * (x-int(width/2))/width
                            y_render = self.field_of_view_y * (y-int(height/2))/height * aspectratio 

                            direction = mathutils.Vector((x_render + dx*(random.random()-0.5), y_render + dy*(random.random()-0.5), -1)) 

                            direction.rotate(rotation)
                            direction = direction.normalized()

                            if False:
                                print("JOLO")
                                sbuf[y,x,0:3] += self.background_hit(direction)#end_dir)
                            else:
                                # cast a ray into the scene  
                                hit, hit_bh, end_dir, end_loc = self.spacetime_ray_cast(depsgraph, origin, direction) #depsgraph.scene.ray_cast(depsgraph, origin, direction)

                                if hit:
                                    sbuf[y,x,0:3] += self.spacetime_hit(depsgraph, lamps, hit, loc, normal, index, ob, mat)# This is wrong
                                else:
                                    if hit_bh and not hit:
                                        #print("BH HIT")
                                        sbuf[y,x,0:3] += np.array([0, 0, 0])
                                    else:
                                        sbuf[y,x,0:3] += self.background_hit(end_dir)
                                #sbuf[y,x,0:3] += self.spacetime_get_pixel(x, y, dx, dy, width, height, depsgraph, origin, rotation)
                    
                        
                    buf[y,:,0:3] = sbuf[y,:,0:3] / (s+1)
                    if y < height-1:  
                        buf[y+1,:,0:3] = 1 - buf[y+1,:,0:3]  
                    
                    # Print some timing information                    
                    if y%50 == 0:
                        ave_blender_functions_times = np.average(np.array(blender_functions_times))
                        ave_ray_trace_times = np.average(np.array(ray_trace_times))
                        blackhole_hit_times = np.average(np.array(blackhole_hit_times))
                        print("        "+"Update | y: ", str(y)+"/"+str(height), " | Ave. row time: ", round(time.time() - row_start, 1), " | Elapsed tot. time: ", round(time.time()-start, 1))

                    yield (s*width*height+width*y)/N

            print("    Sample time: ", round(time.time() - sample_start, 1))
        timeCalc = time.time() - start
        print("  Ray trace timing: ", timeCalc)
        buf.shape = -1,4
        return buf

    # ############################################################################################################################
    #def spacetime_get_pixel(self, x, y, dx, dy, width, height, depsgraph, origin, rotation):
    # ############################################################################################################################
        # # get the direction.  
        # # camera points in -x direction, FOV = 90 degrees  
        # aspectratio = height/width
        # x_render = (x-int(width/2))/width
        # y_render = (y-int(height/2))/height * aspectratio 

        # direction = mathutils.Vector((x_render + dx*(random.random()-0.5), y_render + dy*(random.random()-0.5), -1)) 

        # direction.rotate(rotation)
        # direction = direction.normalized()

        # # cast a ray into the scene  
        # hit, hit_bh, end_dir, end_loc = self.spacetime_ray_cast(depsgraph, origin, direction) #depsgraph.scene.ray_cast(depsgraph, origin, direction)
        

        # if hit:
        #     return self.spacetime_hit(depsgraph, lamps, hit, loc, normal, index, ob, mat)# This is wrong
        # else:
        #     if hit_bh and not hit:
        #         print("BH HIT")
        #         return np.array([0, 0, 0])
        #     else:
        #         return self.background_hit(end_dir)

    # ############################################################################################################################
    def spacetime_ray_cast(self, depsgraph, origin, direction):
    # ############################################################################################################################
        # Origin: camera location
        origin_ray_global = origin
        # Direction: direction from the camera, normalized

        # origin: start location of the ray relative to the blackhole position
        origin = origin - self.bh_loc #mathutils.Vector([0, 0, 0]) # We take the blackhole in the origin. Should become: blackhole_ob.location

        #_start_conditions = [direction[0], origin[0], direction[1], origin[1], direction[2], origin[2]]
        k_x_0, x0, k_y_0, y0, k_z_0, z0 = direction[0], origin[0], direction[1], origin[1], direction[2], origin[2]
        # result = self.GeoInt.calc_trajectory(\
        #                 k_x_0, x0, k_y_0, y0, k_z_0, z0,\
        #                 R_end = 10000,\
        #                 max_step = self.max_integration_step, verbose = False )
        
        k0_xyz = np.array([k_x_0, k_y_0, k_z_0])
        x0_xyz = np.array([x0, y0, z0])
        k_xyz, x_xyz, result = self.GeoInt.calc_trajectory(k0_xyz, x0_xyz, max_step = self.max_integration_step,\
                                                               curve_end=50, nr_points_curve=10000, verbose=False)
        
        if result['start_inside_hole'] == False:
            hit_bh = result["hit_blackhole"]

            x, y, z = x_xyz
            k_x, k_y, k_z = k_xyz
            #k_x, x, k_y, y, k_z, z = result.y
            curve = np.array([x, y, z])

            # NOW YOU DO COLLISION DETECTION
            hit = False

            end_loc = np.array([x[-1], y[-1], z[-1]])
            end_dir = np.array([k_x[-1], k_y[-1], k_z[-1]])
            
            return hit, hit_bh, end_dir, end_loc
        else:
            print("Camera INSIDE blackhole")
            return False, True, [], []


    # ############################################################################################################################
    def spacetime_hit(self, depsgraph, lamps, hit, loc, normal, index, ob, mat, intensity = 10, eps = 1e-5):
    # ############################################################################################################################
        ##
        # This needs some serius work!
        ##

        object_tex_name = self.textures['moon']['texture_name']
        #disk_tex_name = self.textures[self.disk_texture]['texture_name']


        # intensity for all lamps  
        # eps: small offset to prevent self intersection for secondary rays
        
        # the default background is black for now  
        color = np.zeros(3)  
        emission = False
        if hit:  
            if emission:
                hit_norm = loc-ob.location
                hit_norm = hit_norm/np.linalg.norm(hit_norm)
                th = np.arccos(hit_norm[2])
                ph = np.arctan(hit_norm[1]/hit_norm[0])
                color = np.array(bpy.data.textures[object_tex_name].evaluate((ph/(2*np.pi),th/np.pi,0)).xyz)

            else:
                color = np.zeros(3)
                base_color = np.ones(3) * intensity  # light color is white  
                for lamp in lamps:  
                    # for every lamp determine the direction and distance  
                    light_vec = lamp.location - loc  
                    light_dist = light_vec.length_squared  
                    light_dir = light_vec.normalized()  
                     
                    # cast a ray in the direction of the light starting  
                    # at the original hit location  
                    lhit, lloc, lnormal, lindex, lob, lmat = depsgraph.scene.ray_cast(depsgraph, loc+light_dir*eps, light_dir)  
                     
                    # if we hit something we are in the shadow of the light  
                    if not lhit:  
                        # otherwise we add the distance attenuated intensity  
                        # we calculate diffuse reflectance with a pure   
                        # lambertian model  
                        # https://en.wikipedia.org/wiki/Lambertian_reflectance  
                        color += base_color * intensity * normal.dot(light_dir)/light_dist  
            
        return color


    # ############################################################################################################################
    def background_hit(self, direction):
    # ############################################################################################################################
        if self.back_ground_texture_key in self.textures:
            background_tex_name = self.textures[self.back_ground_texture_key]['texture_name']
        else:
            background_tex_name = ""
        #background_tex_name = self.textures['background']['texture_name']
        test_output = False
        
        if test_output == True:
            if direction[2] <= 0:
                return np.array([0,direction[2],direction[1]])
            else:
                return np.array([0,0,direction[2]])
        else:
            color = np.zeros(3)  
            if background_tex_name in bpy.data.textures:
                if direction[2] > 1 or direction[2]<-1:
                    print("Wrong, dir not normalized: ", direction)
                theta = 1-np.arccos(direction[2])/np.pi
                phi = np.arctan2(direction[1], direction[0])/np.pi
                color = np.array(bpy.data.textures[background_tex_name].evaluate(
                                 (-phi,2*theta-1,0)).xyz)
            else:
                color = np.array([0,0,0])
            return color

















    # ############################################################################################################################
    def blackhole_hit(self, SW, approx, depsgraph, direction, buf, lamps, hit, loc, normal, index, ob, mat):
    # ############################################################################################################################
        save_loc = loc
        save_ob = ob
        
        # We will work in coordinates from the center of the blackhole
        loc = loc-ob.location
        
        # We do a ray cast in curved space-time
        if self.approx:
            end_loc, end_dir, mes = self.aSW.generatedRayTracer(loc, direction)
            error_mes = "loc: "+str(loc)+"-"+str(round(np.linalg.norm(loc), 6))+" end_loc: "+str(end_loc)+"-"+str(round(np.linalg.norm(end_loc), 6))
            error_mes += ", end_dir: "+str(end_dir) + "dir: " + str(direction)
        else:
            x_SW, y_SW, z_SW, end_loc, end_dir, mes = self.SW.ray_trace(direction, \
                                                            loc_hit = loc, \
                                                            exit_tolerance = self.exit_tolerance, \
                                                            ratio_obj_to_blackhole = self.ratio_obj_to_blackhole, \
                                                            curve_end = self.SW.approximateCurveEnd(self.ratio_obj_to_blackhole),\
                                                            max_step = self.max_integration_step)
                                                            #curve_end = 50 + 2*50*(self.ratio_obj_to_blackhole/20 -1),\
                                                            #warnings=False)

            # If we hit the disk        
            if self.disk_on:
                disk_info = self.checkHitDisk(x_SW, y_SW, z_SW, self.ratio_obj_to_blackhole, \
                                            R_in=self.disk_R_in*self.ratio_obj_to_blackhole, R_out = self.disk_R_out*self.ratio_obj_to_blackhole)
                if disk_info["hit"]:


                    # !!! NOTE: I AM SKIPPING THE POSSIBILITY THAT THE BACKGROUND COLOR HITS AN OBJECT IN THE BLENDER SCENE
                    # In other words, no transparency
                    # Get background color:
                    if mes['hit_blackhole']:
                        #print( mes['hit_blackhole'], end_loc == [] )
                        back_ground_color = np.array([0, 0, 0])
                    elif 'error' in mes.keys():
                        back_ground_color = np.array([0, 0, 0])
                    else:
                        back_ground_color = np.array([0, 0, 0])#self.background_hit(end_dir)

                    color = disk_info['color']*disk_info['intensity']# + back_ground_color * (1-disk_info['intensity'])

                    return color #disk_info['color'] #np.array([1,1,1])

        if self.mark_x_min != -1 or self.mark_x_max != -1 or self.mark_y_min != -1 or self.mark_y_max != -1:
            self.debug_string += "["+str(list(loc))+", "+str(list(direction))+", "+str(list(end_loc))+", "+str(list(end_dir))+"], "
                    
        # If we hit the blackhole, we can set the pixel to black
        if mes['hit_blackhole']:
            return np.array([0, 0, 0])
        # Some extra error handeling that should not happen
        if 'error' in mes.keys():
            if mes['error'] == 'Outside':
                # Marking pixels red
                return np.array([1,0,0])
            
        # Otherwise, set the end_loc to the global coordinates
        end_loc = end_loc + ob.location
        # And do a new raycast
        hit, loc, normal, index, ob, mat = depsgraph.scene.ray_cast(depsgraph, end_loc, end_dir) 
        
        # If the second ray hits something
        if hit:
            # It should not be the BH again, since it just came out of it
            if "isBH" in ob.data:

                print(np.linalg.norm(save_loc), np.linalg.norm(end_loc), save_loc, direction, end_loc, end_dir)
                if end_dir[2] < 0:
                    return np.array([0, 0, 1])
                else:
                    return np.array([0,1,0])
            # If the second ray makes a proper hit, go on and do a normal hit
            return self.normal_hit(depsgraph, lamps, hit, loc, normal, index, ob, mat)        
        else:
            # If the second ray hits nothing, do a background hit
            return self.background_hit(end_dir)
        
    # ##


    # ############################################################################################################################

    # ############################################################################################################################
    def checkHitDisk(self, x, y, z, ratio, R_in, R_out):
        disk_tex_name = self.textures[self.disk_texture]['texture_name']

        for i in range(len(x)-1):
            if (z[i+1] < 0 and z[i] >= 0) or (z[i+1] > 0 and z[i] <= 0):

                l0 = -z[i]/(z[i+1]-z[i])
                x_disk = x[i]+(x[i+1]-x[i])*l0
                y_disk = y[i]+(y[i+1]-y[i])*l0

                R = np.sqrt(x_disk**2 + y_disk**2)
                if (R <= R_out) and (R >= R_in):


                    scale = (R-R_in)/(R_out-R_in)

                    mean, stddev = self.disk_mean, self.disk_stddev
                    intensity = self.disk_intensity * np.exp(-((scale-mean)**2)/(2*stddev**2)) * 1/(np.sqrt(2*np.pi*stddev)) 

                    texture_x = (self.disk_phase + np.arccos(x_disk/R) * (y_disk/np.absolute(y_disk))) / np.pi #x_disk/R

                    rgb = np.array(bpy.data.textures[disk_tex_name].evaluate( (texture_x, (R-R_in)/(R_out-R_in), 0) ).xyz)

                    return {"hit": True, "loc": np.array([ x_disk, y_disk, 0.0 ]), "color": rgb, "intensity":intensity}

        return {"hit": False}

    # ############################################################################################################################
    def loadTextures(self):
    # ############################################################################################################################
                            
        for key, tex in self.textures.items():
            force = False
            if not tex["file_name"] in bpy.data.images or force:
                print("LOADING:", tex['file_name'])
                bpy.data.images.load(os.path.join(self.texture_dir, tex["file_name"]))
            img = bpy.data.images[tex["file_name"]]
            
            if not tex["texture_name"] in bpy.data.textures or force:
                world_tex = bpy.data.textures.new(tex["texture_name"], "IMAGE")
                world_tex.image = bpy.data.images[tex["file_name"]]#"bg_orion.png"]#bpy.data.images.load("D:/pic.png")
            else:
                world_tex = bpy.data.textures[tex["texture_name"]]



# ############################################################################################################################
# ############################################################################################################################
# 
# ############################################################################################################################
# ############################################################################################################################

# #######################################################

# #######################################################
def get_panels():
     exclude_panels = {
         'VIEWLAYER_PT_filter',
         'VIEWLAYER_PT_layer_passes',
     }

     panels = []
     for panel in bpy.types.Panel.__subclasses__():
         #print(panel)
         if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
             if panel.__name__ not in exclude_panels:
                 panels.append(panel)

     return panels

# #######################################################

# #######################################################
PROPS = [
     ('metric', bpy.props.StringProperty(name='metric', default='schwarzschild')),
     ('blackhole_obj', bpy.props.PointerProperty(name='blackhole_obj', type=bpy.types.Object)),
     ('mass', bpy.props.FloatProperty(name='Mass', default=0.5)),
     #('ratio_obj_to_blackhole', bpy.props.FloatProperty(name='Ratio Object to Blackhole', default=30.0)),
     #('exit_tolerance', bpy.props.FloatProperty(name='Exit Tolerance', default=0.2)),
     ('max_integration_step', bpy.props.FloatProperty(name='max_integration_step', default=10000)),
     ('sampling_seed', bpy.props.FloatProperty(name='sampling_seed', default=42)),
     ('field_of_view_x', bpy.props.FloatProperty(name='field_of_view_x', default=1)),
     ('field_of_view_y', bpy.props.FloatProperty(name='field_of_view_y', default=1)),
     ('disk_on', bpy.props.BoolProperty(name='Disk', default=False)),
     ('disk_R_in', bpy.props.FloatProperty(name='disk_R_in', default=0.15)),
     ('disk_R_out', bpy.props.FloatProperty(name='disk_R_out', default=0.35)),
     ('disk_phase', bpy.props.FloatProperty(name='disk_phase', default=0)),
     ('disk_mean', bpy.props.FloatProperty(name='disk_mean', default=0.2)),
     ('disk_stddev', bpy.props.FloatProperty(name='disk_stddev', default=0.3)),
     ('disk_intensity', bpy.props.FloatProperty(name='disk_intensity', default=1.0)),
     ('approx', bpy.props.BoolProperty(name='Approx', default=False)),
     ('back_ground_texture_key', bpy.props.StringProperty(name='back_ground_texture_key', default="bg_ngc3293")),
     ('disk_texture', bpy.props.StringProperty(name='disk_texture', default="disk_clouds_high_contr")),
     ('mark_y_min', bpy.props.FloatProperty(name='mark_y_min', default=-1.)),
     ('mark_y_max', bpy.props.FloatProperty(name='mark_y_max', default=-1.)),
     ('mark_x_min', bpy.props.FloatProperty(name='mark_x_min', default=-1.)),
     ('mark_x_max', bpy.props.FloatProperty(name='mark_x_max', default=-1.)),
 ]

# #######################################################

# #######################################################
from bpy.types import Panel  
from bl_ui.properties_render import RenderButtonsPanel  
  
class CUSTOM_RENDER_PT_blackhole(RenderButtonsPanel, Panel):  
    bl_label = "Blackhole Settings"  
    COMPAT_ENGINES = {RelativisticRenderEngine.bl_idname}  
  
    def draw_header(self, context):  
        #rd = bpy.context.scene.render
        rd = bpy.context.scene
        #self.layout.prop(rd, "background_image_renderer", text="background_image_renderer")
        #self.layout.prop(rd, "ratio_obj_to_blackhole", text="ratio")
        #self.layout.prop(rd, "use_antialiasing", text="")  
  
    def draw(self, context):  
        layout = self.layout  
  
        rd = bpy.context.scene#.render  
        #layout.active = rd.use_antialiasing  
  
        split = layout.split()  
  
        col = split.column()  
        col.row().prop(rd, "metric", text="Metric")#, expand=True)  
        col.row().prop(rd, "blackhole_obj", text="Blackhole")#, expand=True)  
        col.row().prop(rd, "mass", text="Mass")#, expand=True)  
        #col.row().prop(rd, "ratio_obj_to_blackhole", text="Ratio Object to Blackhole")#, expand=True)  
        #col.row().prop(rd, "exit_tolerance", text="exit_tolerance")#, expand=True)  
        col.row().prop(rd, "max_integration_step", text="max_integration_step")#, expand=True)  
        
        col.row().prop(rd, "field_of_view_x", text="field_of_view_x")#, expand=True)  
        col.row().prop(rd, "field_of_view_y", text="field_of_view_y")#, expand=True)  

        col.row().prop(rd, "sampling_seed", text="Sampling seed")#, expand=True)  
        
        col.row().prop(rd, "disk_on", text="disk_on")
        col.row().prop(rd, "disk_R_in", text="Disk inner radius")
        col.row().prop(rd, "disk_R_out", text="Disk outer radius")
        col.row().prop(rd, "disk_phase", text="Disk phase")
        col.row().prop(rd, "disk_mean", text="Disk mean")
        col.row().prop(rd, "disk_stddev", text="Disk Std.dev.")
        col.row().prop(rd, "disk_intensity", text="Disk intensity")
        
        #col.row().prop(rd, "approx", text="Approximate")
        col.row().prop(rd, "back_ground_texture_key", text="back_ground_texture_key")
        col.row().prop(rd, "disk_texture", text="Disk texture") 
        col.row().prop(rd, "mark_x_min", text="mark_x_min")
        col.row().prop(rd, "mark_x_max", text="mark_x_max")
        col.row().prop(rd, "mark_y_min", text="mark_y_min")
        col.row().prop(rd, "mark_y_max", text="mark_y_max")





def register():
    # Register the RenderEngine
    bpy.utils.register_class(RelativisticRenderEngine)
    bpy.utils.register_class(CUSTOM_RENDER_PT_blackhole)
    #bpy.utils.register_class(EXAMPLE_PT_panel_1)

    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
        #setattr(bpy.types.RenderSettings, prop_name, prop_value)

    for panel in get_panels()+[CUSTOM_RENDER_PT_blackhole]:
        panel.COMPAT_ENGINES.add(RelativisticRenderEngine.bl_idname)#'gr_ray_tracer')#CUSTOM')
    
    from bl_ui import (
            properties_render,
            properties_material,
#            properties_data_lamp,
            properties_world,
#            properties_texture,
            )

    from cycles import(ui)#  cycles.ui.CYCLES_WORLD_PT_settings_surface
    properties_render.RENDER_PT_eevee_sampling.COMPAT_ENGINES.add(RelativisticRenderEngine.bl_idname)
    properties_world.WORLD_PT_context_world.COMPAT_ENGINES.add(RelativisticRenderEngine.bl_idname)
    #properties_world.WORLD_PT_environment_lighting.COMPAT_ENGINES.add(RelativisticRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_context_material.COMPAT_ENGINES.add(RelativisticRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_surface.COMPAT_ENGINES.add(RelativisticRenderEngine.bl_idname)
    #properties_material.MATERIAL_PT_diffuse.COMPAT_ENGINES.add(RelativisticRenderEngine.bl_idname)

def unregister():
    bpy.utils.unregister_class(RelativisticRenderEngine)
    bpy.utils.unregister_class(CUSTOM_RENDER_PT_blackhole)

    #bpy.utils.unregister_class(EXAMPLE_PT_panel_1)
    
    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)
        
    for panel in get_panels()+[CUSTOM_RENDER_PT_blackhole]:
        if RelativisticRenderEngine.bl_idname in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove(RelativisticRenderEngine.bl_idname)#'gr_ray_tracer')#CUSTOM')

if __name__ == "__main__":
    register()

