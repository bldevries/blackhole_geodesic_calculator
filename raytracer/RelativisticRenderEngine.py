

from importlib import reload
import bpy
import mathutils
import numpy as np
import curvedpy
import os
import time
import random
#reload(curvedpy)

from bpy.types import Panel  
from bl_ui.properties_render import RenderButtonsPanel  

bl_info = {
    "name": "Relativistic Render Engine",
    "bl_label": "Relativistic Render Engine",
    "blender": (4, 1, 0),
    "category": "Render",
}

##################################################################################################################################################
##################################################################################################################################################
# CLASS: RelativisticRenderEngine
##################################################################################################################################################
##################################################################################################################################################
class RelativisticRenderEngine(bpy.types.RenderEngine):
    # Inspired by: https://docs.blender.org/api/current/bpy.types.RenderEngine.html?highlight=renderengine
    bl_idname = "RelRenEn"
    bl_label = "Relativistic"
    bl_use_preview = True
    
    # GOod sky image to use
    #//../Dropbox/0_DATA_BEN/PHYSICS/PROJECTS/blackhole_geodesic_calculator/raytracer/textures/high_ngc3293_eso_8682×8436.jpg

    # #########################################################################
    def render(self, depsgraph):
    # #########################################################################
        print("Starting Render")

        #########################
        # Integration properties
        #########################
        self.max_integration_step = depsgraph.scene.max_integration_step
        self.sampling_seed = depsgraph.scene.sampling_seed
        if self.max_integration_step == -1:
            self.max_integration_step = np.inf 
        self.int_depth_curve_end = depsgraph.scene.integration_depth
        

        #########################
        # Render settings
        #########################
        self.samples = bpy.data.scenes['Scene'].eevee.taa_render_samples
        self.scale = depsgraph.scene.render.resolution_percentage / 100.0
        self.res_x = int(depsgraph.scene.render.resolution_x*self.scale)
        self.res_y = int(depsgraph.scene.render.resolution_y*self.scale)
        
        self.field_of_view_x = depsgraph.scene.field_of_view_x
        self.field_of_view_y = depsgraph.scene.field_of_view_y
        
        #########################
        # LOADING SKY TEXTURE
        #########################
        self.sky_image_path = depsgraph.scene.sky_image
        sky_image_filename = os.path.basename(self.sky_image_path)
        self.sky_tex_name = sky_image_filename + "_tex"
        print(sky_image_filename)
        if not sky_image_filename in bpy.data.images:
            print("LOADING:", sky_image_filename)
            bpy.data.images.load(self.sky_image_path)
        
        if not self.sky_tex_name in bpy.data.textures:
            world_tex = bpy.data.textures.new(self.sky_tex_name, "IMAGE")
            world_tex.image = bpy.data.images[sky_image_filename]
        else:
            world_tex = bpy.data.textures[self.sky_tex_name]

        #########################
        # Blackhole properties
        #########################
        self.mass = depsgraph.scene.mass # R_horizon = 2*M in Geometrized units
        if depsgraph.scene.blackhole_obj == None:
            self.bh_loc = mathutils.Vector([0, 0, 0])
        else:
            self.bh_loc = depsgraph.scene.blackhole_obj.location


        #########################
        # NEED TO GO!
        #########################
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
        


        #########################
        # Print all settings
        #########################
        print("  - Integr. step: ", self.max_integration_step)
        # print("  - Metric: ", self.metric)
        print("  - Mass: ", self.mass)
        print("  - BH location: ", self.bh_loc)
        print("  - Sampling seed: ", self.sampling_seed)
        print("  - Samples: ", self.samples)
        print("  - BG Texture: ", self.sky_tex_name)

        #########################
        # Initiate the geodesic solver
        #########################
        self.GeoInt = curvedpy.GeodesicIntegratorSchwarzschild(mass = self.mass, time_like = False, verbose=False)


        #########################
        # For some reason I need to update the depsgraph otherwise things wont render 
        # from the console properly using "-f <frame_nr>"
        # This might be a horrible hack :S
        depsgraph = bpy.context.evaluated_depsgraph_get() 
        depsgraph.update()

        #########################
        # Start the rendering
        #########################
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
                                                               curve_end=self.int_depth_curve_end, nr_points_curve=10000, verbose=False)
        
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
    
        color = np.zeros(3)  
        if self.sky_tex_name in bpy.data.textures:
            if direction[2] > 1 or direction[2]<-1:
                print("Wrong, dir not normalized: ", direction)
            theta = 1-np.arccos(direction[2])/np.pi
            phi = np.arctan2(direction[1], direction[0])/np.pi
            color = np.array(bpy.data.textures[self.sky_tex_name].evaluate( (-phi,2*theta-1,0) ).xyz )
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
        

##################################################################################################################################################
##################################################################################################################################################
# CLASS/PANEL CUSTOM_RENDER_PT_blackhole
##################################################################################################################################################
##################################################################################################################################################

class CUSTOM_RENDER_PT_blackhole(RenderButtonsPanel, Panel):  
    bl_label = "Blackhole Settings"  
    COMPAT_ENGINES = {RelativisticRenderEngine.bl_idname}  
  
    def draw_header(self, context):  
        rd = bpy.context.scene
  
    def draw(self, context):  
        layout = self.layout  
  
        rd = bpy.context.scene#.render  
        #layout.active = rd.use_antialiasing  
  
        split = layout.split()  
  
        col = split.column()  
        col.row().prop(rd, "blackhole_obj", text="Blackhole")#, expand=True)  
        col.row().prop(rd, "mass", text="Mass")#, expand=True)  
        col.row().prop(rd, "max_integration_step", text="Max integration step")#, expand=True)  
        col.row().prop(rd, "integration_depth", text="Integration depth")#, expand=True)  
        col.row().prop(rd, "field_of_view_x", text="field_of_view_x")#, expand=True)  
        col.row().prop(rd, "field_of_view_y", text="field_of_view_y")#, expand=True)  

        col.row().prop(rd, "sampling_seed", text="Sampling seed")#, expand=True)  
        col.row().prop(rd, "sky_image", text="Sky image")
        
        col.row().prop(rd, "mark_x_min", text="mark_x_min")
        col.row().prop(rd, "mark_x_max", text="mark_x_max")
        col.row().prop(rd, "mark_y_min", text="mark_y_min")
        col.row().prop(rd, "mark_y_max", text="mark_y_max")


##################################################################################################################################################
##################################################################################################################################################
# ADDON FUNCTIONS
##################################################################################################################################################
##################################################################################################################################################

PROPS = [
     ('blackhole_obj', bpy.props.PointerProperty(name='blackhole_obj', type=bpy.types.Object)),
     ('mass', bpy.props.FloatProperty(name='Mass', default=0.5)),
     ('max_integration_step', bpy.props.FloatProperty(name='max_integration_step', default=10000)),
     ('integration_depth', bpy.props.FloatProperty(name='integration_depth', default=50)),
     ('sampling_seed', bpy.props.FloatProperty(name='sampling_seed', default=42)),
     ('field_of_view_x', bpy.props.FloatProperty(name='field_of_view_x', default=1)),
     ('field_of_view_y', bpy.props.FloatProperty(name='field_of_view_y', default=1)),
     ('sky_image', bpy.props.StringProperty(name='sky_image', default="", subtype="FILE_PATH")),
     ('mark_y_min', bpy.props.FloatProperty(name='mark_y_min', default=-1.)),
     ('mark_y_max', bpy.props.FloatProperty(name='mark_y_max', default=-1.)),
     ('mark_x_min', bpy.props.FloatProperty(name='mark_x_min', default=-1.)),
     ('mark_x_max', bpy.props.FloatProperty(name='mark_x_max', default=-1.)),
 ]

# ############################################################################################################################
def get_panels():
# ############################################################################################################################
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

# ############################################################################################################################
def register():
# ############################################################################################################################
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

# ############################################################################################################################
def unregister():
# ############################################################################################################################
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

