

from importlib import reload
import bpy
import mathutils
import numpy as np
import curvedpy
import os
import time
reload(curvedpy)


class CustomRenderEngine(bpy.types.RenderEngine):
    # Inspired by: https://docs.blender.org/api/current/bpy.types.RenderEngine.html?highlight=renderengine
    bl_idname = "gr_ray_tracer"
    bl_label = "General Relativistic Renderer"
    bl_use_preview = True
    
    ratio_obj_to_blackhole = 30
    exit_tolerance = 0.02
    aSW = curvedpy.ApproxSchwarzschildGeodesic(ratio_obj_to_blackhole = ratio_obj_to_blackhole, \
                                                            exit_tolerance = exit_tolerance)

    # ############################################################################################################################
    def render(self, depsgraph):
    # ############################################################################################################################



        self.textures = self.loadTextures()

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
        
        start = time.time()
        buf = self.ray_trace(depsgraph, res_x, res_y)
        timeCalc = time.time() - start
        print("Ray trace timing: ", timeCalc)
        buf.shape = -1,4

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, res_x, res_y)#self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = buf.tolist()
        self.end_result(result)


    # ############################################################################################################################
    def ray_trace(self, depsgraph, width, height, approx = True):
    # ############################################################################################################################
        
        # SW is the Schwarzschild metric class that we use to do a ray cast in curved space-time
        # We can use an approximation for testing
        if approx:
            SW = self.aSW
        # Or the real deal for proper renders
        else:
            SW = curvedpy.SchwarzschildGeodesic()   

        # We collect the light objects in our scene            
        lamps = [ob for ob in depsgraph.scene.objects if ob.type == 'LIGHT']  

        # create a buffer to store the calculated intensities  
        buf = np.ones(width*height*4)
        buf.shape = height,width,4
        
        # This is the location and rotation of the camera
        origin = depsgraph.scene.camera.location  
        rotation = depsgraph.scene.camera.rotation_euler  
        
        # loop over all pixels once (no multisampling yet)  
        for y in range(height):  
            for x in range(width):
                # get the direction.  
                # camera points in -x direction, FOV = 90 degrees  
                x_render, y_render = (x-int(width/2))/width, (y-int(height/2))/height
                direction = mathutils.Vector((x_render, y_render, -1))
                direction.rotate(rotation)
                 
                # cast a ray into the scene  
                hit, loc, normal, index, ob, mat = depsgraph.scene.ray_cast(depsgraph, origin, direction)

                if hit:
                    if "isBH" in ob.data:
                        buf[y,x,0:3] = self.blackhole_hit(SW, approx, depsgraph, direction, buf, lamps, hit, loc, normal, index, ob, mat)
                    else:
                        buf[y,x,0:3] = self.normal_hit(depsgraph, lamps, hit, loc, normal, index, ob, mat)
                else:
                    buf[y,x,0:3] = self.background_hit(direction)
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
                                                            curve_end = 50 + 2*50*(self.ratio_obj_to_blackhole/20 -1),\
                                                            warnings=False)

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

                print(error_mes)
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
        
        background_tex_name = self.textures['background']['texture_name']
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
    def loadTextures(texture_dir ='/Users/vries001/Dropbox/0_DATA_BEN/PHYSICS/PROJECTS/relativistic_raystracer/images/'):
    # ############################################################################################################################
        
        textures = {"background": {"file_name": "8k_stars_milky_way.jpg", "texture_name": "8k_stars_milky_way.jpg"},\
                    "moon": {"file_name": "8k_moon.jpg", "texture_name": "8k_moon.jpg"}}
        for key, tex in textures.items():
            if not tex["file_name"] in bpy.data.images:
                print("Loading: ", tex["file_name"])
                bpy.data.images.load(os.path.join(texture_dir, tex["file_name"]))
            img = bpy.data.images[tex["file_name"]]
            
            if not tex["texture_name"] in bpy.data.textures:
                print("LoadingL ", tex["texture_name"])
                world_tex = bpy.data.textures.new(tex["texture_name"], "IMAGE")
                world_tex.image = bpy.data.images[tex["file_name"]]#"bg_orion.png"]#bpy.data.images.load("D:/pic.png")
            else:
                world_tex = bpy.data.textures[tex["texture_name"]]

        return textures

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

PROPS = [
     ('background_image_renderer', bpy.props.StringProperty(name='File name', default='...', subtype='FILE_NAME')),

 ]


def register():
    # Register the RenderEngine
    bpy.utils.register_class(CustomRenderEngine)
    #bpy.utils.register_class(EXAMPLE_PT_panel_1)

    #for (prop_name, prop_value) in PROPS:
    #    setattr(bpy.types.Scene, prop_name, prop_value)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('gr_ray_tracer')#CUSTOM')
    
    from bl_ui import (
            properties_render,
            properties_material,
#            properties_data_lamp,
            properties_world,
#            properties_texture,
            )

    from cycles import(ui)#  cycles.ui.CYCLES_WORLD_PT_settings_surface
    properties_world.WORLD_PT_context_world.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    #properties_world.WORLD_PT_environment_lighting.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_context_material.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    properties_material.EEVEE_MATERIAL_PT_surface.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)
    #properties_material.MATERIAL_PT_diffuse.COMPAT_ENGINES.add(CustomRenderEngine.bl_idname)

def unregister():
    bpy.utils.unregister_class(CustomRenderEngine)
    #bpy.utils.unregister_class(EXAMPLE_PT_panel_1)
    
    #for (prop_name, _) in PROPS:
    #    delattr(bpy.types.Scene, prop_name)
        
    for panel in get_panels():
        if 'gr_ray_tracer' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('gr_ray_tracer')#CUSTOM')

if __name__ == "__main__":
    register()
