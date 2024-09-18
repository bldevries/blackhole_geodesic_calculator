

from importlib import reload
import bpy
import mathutils
import numpy as np
import curvedpy
import os
import time
import random
reload(curvedpy)

bl_info = {
    "name": "gr_ray_tracer_2",
    "bl_label": "gr render",
    "blender": (4, 1, 0),
    "category": "Render",
}

class CustomRenderEngine(bpy.types.RenderEngine):
    # Inspired by: https://docs.blender.org/api/current/bpy.types.RenderEngine.html?highlight=renderengine
    bl_idname = "gr_ray_tracer"
    bl_label = "General Relativistic Renderer"
    bl_use_preview = True
    # bl_info
    
    textures = {"background": {"file_name": "8k_stars_milky_way.jpg", "texture_name": "8k_stars_milky_way.jpg"},\
                "bg_ngc3293": {"file_name": "high_ngc3293_eso_8682×8436.jpg", "texture_name": "high_ngc3293_eso_8682×8436.jpg"},\
                "high_PIA23647": {"file_name": "high_PIA23647.png", "texture_name": "high_PIA23647.png"},\
                "perseus-cluster": {"file_name": "high_1-Perseus-cluster_1oEasJg_6500×6500.jpg", "texture_name": "high_1-Perseus-cluster_1oEasJg_6500×6500.jpg"},\
                "moon": {"file_name": "8k_moon.jpg", "texture_name": "8k_moon.jpg"},\
                "test": {"file_name": "test.png", "texture_name": "test"}}
    texture_dir ='/Users/vries001/Dropbox/0_DATA_BEN/PHYSICS/PROJECTS/blackhole_geodesic_calculator/raytracer/textures/'

    # ############################################################################################################################
    def render(self, depsgraph):
    # ############################################################################################################################
        print("Starting Render")
        self.ratio_obj_to_blackhole = depsgraph.scene.ratio_obj_to_blackhole
        self.exit_tolerance = depsgraph.scene.exit_tolerance
        self.max_integration_step = depsgraph.scene.max_integration_step
        self.disk_on = depsgraph.scene.disk_on
        self.back_ground_texture_key = depsgraph.scene.back_ground_texture_key
        self.metric = depsgraph.scene.metric

        if self.max_integration_step == -1:
            self.max_integration_step = np.inf 

        self.debug_string = ""

        print("  - ", self.ratio_obj_to_blackhole)
        print("  - ", self.exit_tolerance)
        print("  - ", self.disk_on)
        print("  - ", self.back_ground_texture_key)
        print("  - ", self.max_integration_step)

        if self.disk_on and self.approx:
            print("WARNING: disk and approx are both on but do not work together. The disk is turned of.")
            self.disk_on = False

        if self.approx:
            aSW = curvedpy.ApproxSchwarzschildGeodesic(ratio_obj_to_blackhole = ratio_obj_to_blackhole, \
                                                                   exit_tolerance = exit_tolerance)
        else:
            aSW = None


        self.loadTextures()
        # For some reason I need to update the depsgraph otherwise things wont render 
        # from the console properly using "-f <frame_nr>"
        depsgraph = bpy.context.evaluated_depsgraph_get() 


        if self.is_preview:  # we might differentiate later
            pass             # for now ignore completely
        else:
            self.render_scene(depsgraph)#self.tex_name_bg, self.tex_name_stars)

    # ############################################################################################################################
    def render_scene(self, depsgraph):
    # ############################################################################################################################
        scale = depsgraph.scene.render.resolution_percentage / 100.0
        res_x = int(depsgraph.scene.render.resolution_x*scale)
        res_y = int(depsgraph.scene.render.resolution_y*scale)
        
        self.mark_y_min = depsgraph.scene.mark_y_min
        self.mark_y_max = depsgraph.scene.mark_y_max
        self.mark_x_min = depsgraph.scene.mark_x_min
        self.mark_x_max = depsgraph.scene.mark_x_max

        if self.mark_y_min == -1:
            self.mark_y_min = 0
        if self.mark_y_max == -1:
            self.mark_y_max = res_y

        if self.mark_x_min == -1:
            self.mark_x_min = 0
        if self.mark_x_max == -1:
            self.mark_x_max = res_y

        self.mark_y_min = self.mark_y_min * res_y
        self.mark_y_max = self.mark_y_max * res_y
        self.mark_x_min = self.mark_x_min * res_x
        self.mark_x_max = self.mark_x_max * res_x


        width, height = res_x, res_y
        buf = np.ones(width*height*4)
        buf.shape = height,width,4

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, res_x, res_y)#self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]

        samples = bpy.data.scenes['Scene'].eevee.taa_render_samples
        #depsgraph.scene.render.eevee.taa_render_samples
        #int(depsgraph.scene.render.antialiasing_samples) if depsgraph.scene.render.use_antialiasing else 1

        for y in self.ray_trace(depsgraph, width, height, 1, buf, samples): 
            buf.shape = -1,4  
            layer.rect = buf.tolist()  
            self.update_result(result)  
            buf.shape = height,width,4  
            self.update_progress(y/height)  
          
        self.end_result(result) 


        # layer.rect = buf.tolist()
        # self.end_result(result)


    # ############################################################################################################################
    def ray_trace(self, depsgraph, width, height, depths, buf, samples, approx = False):
    # ############################################################################################################################
        
        # SW is the Schwarzschild metric class that we use to do a ray cast in curved space-time
        # We can use an approximation for testing
        if approx:
            SW = self.aSW
        # Or the real deal for proper renders
        else:
            SW = curvedpy.SchwarzschildGeodesic(metric=self.metric)#flat')   #schwarzschild

        # We collect the light objects in our scene            
        lamps = [ob for ob in depsgraph.scene.objects if ob.type == 'LIGHT']  

        # create a buffer to store the calculated intensities  
        # buf = np.ones(width*height*4)
        # buf.shape = height,width,4

        # Buffer for the sample?
        sbuf = np.zeros(width*height*4)  
        sbuf.shape = height,width,4
        
        # This is the location and rotation of the camera
        origin = depsgraph.scene.camera.location  
        rotation = depsgraph.scene.camera.rotation_euler  
        
        aspectratio = height/width
        # Small deviations for the multisampling
        dy = aspectratio/height  
        dx = 1/width  
        random.seed(42)  

        N = samples*width*height
        print("SAMPLING: samples:", samples, ", #rays total: ", N)

        start = time.time()
        for s in range(samples):
            print("  Sample #", s)
            sample_start = time.time()
        # loop over all pixels once (no multisampling yet)  
            for y in range(height):
                if y >= self.mark_y_min and y <= self.mark_y_max:
                    y_render = (y-int(height/2))/height * aspectratio 
                    if y%50 == 0:
                        print("    Progress y: ", y, height)
                    for x in range(width):
                        if x >= self.mark_x_min and x <= self.mark_x_max:

                            # get the direction.  
                            # camera points in -x direction, FOV = 90 degrees  
                            x_render = (x-int(width/2))/width

                            direction = mathutils.Vector((x_render + dx*(random.random()-0.5), y_render + dy*(random.random()-0.5), -1))  
                            
                            direction.rotate(rotation)
                            direction = direction.normalized()

                            # cast a ray into the scene  
                            hit, loc, normal, index, ob, mat = depsgraph.scene.ray_cast(depsgraph, origin, direction)

                            if hit:
                                if "isBH" in ob.data:
                                    sbuf[y,x,0:3] += self.blackhole_hit(SW, approx, depsgraph, direction, buf, lamps, hit, loc, normal, index, ob, mat)
                                else:
                                    sbuf[y,x,0:3] += self.normal_hit(depsgraph, lamps, hit, loc, normal, index, ob, mat)
                            else:
                                sbuf[y,x,0:3] += self.background_hit(direction)
                        
                    buf[y,:,0:3] = sbuf[y,:,0:3] / (s+1)
                    if y < height-1:  
                        buf[y+1,:,0:3] = 1 - buf[y+1,:,0:3]  
                    yield (s*width*height+width*y)/N
            print("    time: ", round(time.time() - sample_start, 1))
        timeCalc = time.time() - start
        print("  Ray trace timing: ", timeCalc)
        # print("######")
        # print(self.debug_string)
        # print("######")
        buf.shape = -1,4
        return buf

    # ############################################################################################################################
    def blackhole_hit(self, SW, approx, depsgraph, direction, buf, lamps, hit, loc, normal, index, ob, mat):
    # ############################################################################################################################
        save_loc = loc
        save_ob = ob
        
        # We will work in coordinates from the center of the blackhole
        loc = loc-ob.location
        
        # We do a ray cast in curved space-time
        # Either we do an approximated ray cast for texting purposes
        if approx:
            end_loc, end_dir, mes = SW.generatedRayTracer(loc, direction)
            error_mes = "loc: "+str(loc)+"-"+str(round(np.linalg.norm(loc), 6))+" end_loc: "+str(end_loc)+"-"+str(round(np.linalg.norm(end_loc), 6))
            error_mes += ", end_dir: "+str(end_dir) + "dir: " + str(direction)
        # Or the real ray cast
        else:
            x_SW, y_SW, z_SW, end_loc, end_dir, mes = SW.ray_trace(direction, \
                                                            loc_hit = loc, \
                                                            exit_tolerance = self.exit_tolerance, \
                                                            ratio_obj_to_blackhole = self.ratio_obj_to_blackhole, \
                                                            curve_end = SW.approximateCurveEnd(self.ratio_obj_to_blackhole),\
                                                            max_step = self.max_integration_step)
                                                            #curve_end = 50 + 2*50*(self.ratio_obj_to_blackhole/20 -1),\
                                                            #warnings=False)
        if self.mark_x_min != -1 or self.mark_x_max != -1 or self.mark_y_min != -1 or self.mark_y_max != -1:
            self.debug_string += "["+str(list(loc))+", "+str(list(direction))+", "+str(list(end_loc))+", "+str(list(end_dir))+"], "

        
        # if(np.linalg.norm(end_loc) < np.linalg.norm(loc)):
        #     print("Exit does not work!!!!")

        def checkHitDisk(x, y, z, ratio, R_in, R_out):
            #R_in, R_out = 0.15*ratio, 0.8*ratio
            #print(R_in, R_out, x[0], y[0], z[0])
            for i in range(len(x)-1):
                if (z[i+1] < 0 and z[i] >= 0) or (z[i+1] > 0 and z[i] <= 0):

                    l0 = -z[i]/(z[i+1]-z[i])
                    x_disk = x[i]+(x[i+1]-x[i])*l0
                    y_disk = y[i]+(y[i+1]-y[i])*l0

                    R = np.sqrt(x_disk**2 + y_disk**2)
                    if (R <= R_out) and (R >= R_in):


                        scale = (R-R_in)/(R_out-R_in)
                        rgb = np.array([1, 0.3 + (1-scale)*(0.9-0.3)  ,0.0])
                        return {"hit": True, "loc": np.array([ x_disk, y_disk, 0.0 ]), "color": rgb}



                    # if ((x[i+1]**2 + y[i+1]**2 >= R_in**2) and (x[i+1]**2 + y[i+1]**2 <= R_out**2)) or \
                    #     ((x[i]**2 + y[i]**2 >= R_in**2) and (x[i]**2 + y[i]**2 <= R_out**2)):
                    #     return {"hit": True, "loc": np.array([ (x[i+1]+x[i])/2, (y[i+1]+y[i])/2, (z[i+1]+z[i])/2 ])}

            return {"hit": False}
        
        # If we hit the disk        
        if self.disk_on:
            disk_info = checkHitDisk(x_SW, y_SW, z_SW, self.ratio_obj_to_blackhole, R_in=0.15*self.ratio_obj_to_blackhole, R_out = 0.35*self.ratio_obj_to_blackhole)
            if disk_info["hit"]:
                return disk_info['color'] #np.array([1,1,1])
                    
        # If we hit the blackhole, we can set the pixel to black
        if mes['hit_blackhole']:
            #print( mes['hit_blackhole'], end_loc == [] )
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
        
#        if ob != None:
#            if ob.name == 'Sphere':#.data["isBH"]:
#                hit = False

    # ############################################################################################################################
    def normal_hit(self, depsgraph, lamps, hit, loc, normal, index, ob, mat, intensity = 10, eps = 1e-5, ):
    # ############################################################################################################################
        
        object_tex_name = self.textures['moon']['texture_name']

        # intensity for all lamps  
        # eps: small offset to prevent self intersection for secondary rays
        
        # the default background is black for now  
        color = np.zeros(3)  
        emission = True
        if hit:  
            if emission:
                #color = np.array([1,1,1])*0.2
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
        
        background_tex_name = self.textures[self.back_ground_texture_key]['texture_name']
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
            return color

    # ############################################################################################################################
    def loadTextures(self):
    # ############################################################################################################################
                            
        for key, tex in self.textures.items():
            if not tex["file_name"] in bpy.data.images:
                #print("Loading image: ", tex["file_name"])
                #print(self.texture_dir, tex["file_name"])
                bpy.data.images.load(os.path.join(self.texture_dir, tex["file_name"]))
            img = bpy.data.images[tex["file_name"]]
            
            if not tex["texture_name"] in bpy.data.textures:
                #print("Loading texture: ", tex["texture_name"])
                world_tex = bpy.data.textures.new(tex["texture_name"], "IMAGE")
                world_tex.image = bpy.data.images[tex["file_name"]]#"bg_orion.png"]#bpy.data.images.load("D:/pic.png")
            else:
                world_tex = bpy.data.textures[tex["texture_name"]]

        #return textures


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
     ('ratio_obj_to_blackhole', bpy.props.FloatProperty(name='Ratio Object to Blackhole', default=30.0)),
     ('exit_tolerance', bpy.props.FloatProperty(name='Exit Tolerance', default=0.2)),
     ('max_integration_step', bpy.props.FloatProperty(name='max_integration_step', default=10000)),
     ('disk_on', bpy.props.BoolProperty(name='Disk', default=False)),
     ('back_ground_texture_key', bpy.props.StringProperty(name='back_ground_texture_key', default="bg_ngc3293")),
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
    COMPAT_ENGINES = {CustomRenderEngine.bl_idname}  
  
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
        col.row().prop(rd, "ratio_obj_to_blackhole", text="Ratio Object to Blackhole")#, expand=True)  
        col.row().prop(rd, "exit_tolerance", text="exit_tolerance")#, expand=True)  
        col.row().prop(rd, "max_integration_step", text="max_integration_step")#, expand=True)  
        col.row().prop(rd, "disk_on", text="disk_on")
        col.row().prop(rd, "back_ground_texture_key", text="back_ground_texture_key")
        col.row().prop(rd, "mark_x_min", text="mark_x_min")
        col.row().prop(rd, "mark_x_max", text="mark_x_max")
        col.row().prop(rd, "mark_y_min", text="mark_y_min")
        col.row().prop(rd, "mark_y_max", text="mark_y_max")





def register():
    # Register the RenderEngine
    bpy.utils.register_class(CustomRenderEngine)
    bpy.utils.register_class(CUSTOM_RENDER_PT_blackhole)
    #bpy.utils.register_class(EXAMPLE_PT_panel_1)

    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
        #setattr(bpy.types.RenderSettings, prop_name, prop_value)

    for panel in get_panels()+[CUSTOM_RENDER_PT_blackhole]:
        panel.COMPAT_ENGINES.add('gr_ray_tracer')#CUSTOM')
    
    from bl_ui import (
            properties_render,
            properties_material,
#            properties_data_lamp,
            properties_world,
#            properties_texture,
            )

    from cycles import(ui)#  cycles.ui.CYCLES_WORLD_PT_settings_surface
    properties_render.RENDER_PT_eevee_sampling.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    properties_world.WORLD_PT_context_world.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    #properties_world.WORLD_PT_environment_lighting.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_context_material.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_surface.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    #properties_material.MATERIAL_PT_diffuse.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)

def unregister():
    bpy.utils.unregister_class(CustomRenderEngine)
    bpy.utils.unregister_class(CUSTOM_RENDER_PT_blackhole)

    #bpy.utils.unregister_class(EXAMPLE_PT_panel_1)
    
    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)
        
    for panel in get_panels()+[CUSTOM_RENDER_PT_blackhole]:
        if 'gr_ray_tracer' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('gr_ray_tracer')#CUSTOM')

if __name__ == "__main__":
    register()

