

# (base) vries001@M22D64KVHFNMV bin % pwd
# /Applications/Blender_versions/Blender4.1.app/Contents/Resources/4.1/python/bin
# (base) vries001@M22D64KVHFNMV bin % ./python3.11 -m pip install ~/Dropbox/0_DATA_BEN/PHYSICS/PROJECTS/blackhole_geodesic_calculator/python_pkg_dev/ -U


bl_info = {
    "name": "general_relativistic_ray_tracer",
    "author": "B.L. de Vries",
    "version": (0, 0, 202407100),
    "blender": (4, 1, 0),
    "location": "",
    "description": "",
    "warning": "",
    "wiki_url": "",
    "category": "Render",
}

from importlib import reload
import bpy
import mathutils
import numpy as np
import curvedpy
reload(curvedpy)

scene = bpy.context.scene


def loadTexture(image_name = "Untitled", tex_name = "MP.001"):
    print("Loading texture: ", image_name, tex_name)
    
    if not image_name in bpy.data.images:

        filename = '/Users/vries001/Dropbox/0_DATA_BEN/PHYSICS/PROJECTS/relativistic_raystracer/images/'+image_name#bg_orion.png'
        print("Loading new image", filename)
        bpy.data.images.load(filename)
    img = bpy.data.images[image_name]
    
    if not tex_name in bpy.data.textures:
        #print(bpy.data.images["Untitled"].pixels[100])
        world_tex = bpy.data.textures.new(tex_name, "IMAGE")
        world_tex.image = bpy.data.images[image_name]#"bg_orion.png"]#bpy.data.images.load("D:/pic.png")
    else:
        world_tex = bpy.data.textures[tex_name]

def blackhole_hit(SW, approx, depsgraph, dir, background, object_tex, buf, lamps, hit, loc, normal, index, ob, mat):
    
    save_loc = loc
    save_ob = ob
    
    loc = loc-ob.location
    
    if approx:
        end_loc, end_dir, mes = SW.generatedRayTracer(loc, dir)
    else:
        ratio_obj_to_blackhole = 60
        x_SW, y_SW, z_SW, end_loc, end_dir, _ = SW.ray_trace(dir, \
                                                        loc_hit = loc, \
                                                        exit_tolerance = 0.1, \
                                                        ratio_obj_to_blackhole = ratio_obj_to_blackhole, \
                                                        curve_end = 50 + 2*50*(ratio_obj_to_blackhole/20 -1),\
                                                        warnings=False)
    if end_loc == []:
        return np.array([0, 0, 0])
    
    end_loc = end_loc + ob.location
    hit, loc, normal, index, ob, mat = scene.ray_cast(depsgraph, end_loc, end_dir) 
    
    if hit:
        if "isBH" in ob.data:

            print(np.linalg.norm(save_loc-save_ob.location), np.linalg.norm(end_loc-ob.location), end_dir[2])
            if end_dir[2] < 0:
                return np.array([0, 0, 1])
            else:
                return np.array([0,1,0])
        return normal_hit(depsgraph, object_tex, lamps, hit, loc, normal, index, ob, mat)        
    else:
        return background_hit(end_dir, background)
    
    if ob != None:
        if ob.name == 'Sphere':#.data["isBH"]:
            hit = False

def normal_hit(depsgraph, object_tex, lamps, hit, loc, normal, index, ob, mat, intensity = 10, eps = 1e-5, ):
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
            color = np.array(bpy.data.textures[object_tex].evaluate((ph/(2*np.pi),th/np.pi,0)).xyz)
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
                lhit, lloc, lnormal, lindex, lob, lmat = scene.ray_cast(depsgraph, loc+light_dir*eps, light_dir)  
                 
                # if we hit something we are in the shadow of the light  
                if not lhit:  
                    # otherwise we add the distance attenuated intensity  
                    # we calculate diffuse reflectance with a pure   
                    # lambertian model  
                    # https://en.wikipedia.org/wiki/Lambertian_reflectance  
                    color += base_color * intensity * normal.dot(light_dir)/light_dist  
        
    return color

def background_hit(dir, background):
    test_output = False
    
    if test_output == True:
        if dir[2] <= 0:
            return np.array([0,dir[2],dir[1]])
        else:
            return np.array([0,0,dir[2]])
    else:
        color = np.zeros(3)  
        if background in bpy.data.textures:
            if dir[2] > 1 or dir[2]<-1:
                print("Wrong, dir not normalized: ", dir)
            theta = 1-np.arccos(dir[2])/np.pi
            phi = np.arctan2(dir[1], dir[0])/np.pi
            color = np.array(bpy.data.textures[background].evaluate(
                             (-phi,2*theta-1,0)).xyz)
        return color
                
def ray_trace(depsgraph, width, height, background, object_tex, approx = False):
    
    print("Background: ", background)
    
    if approx:
        SW = curvedpy.ApproxSchwarzschildGeodesic()
    else:
        SW = curvedpy.SchwarzschildGeodesic()
        
    lamps = [ob for ob in scene.objects if ob.type == 'LIGHT']  

#    intensity = 10  # intensity for all lamps  
#    eps = 1e-5      # small offset to prevent self intersection for secondary rays  

    # create a buffer to store the calculated intensities  
    buf = np.ones(width*height*4)
    buf.shape = height,width,4

    
    # the location of our virtual camera  
    # (we do NOT use any camera that might be present)  

    origin = depsgraph.scene.camera.location  
    rotation = depsgraph.scene.camera.rotation_euler  
    
    # loop over all pixels once (no multisampling)  
    for y in range(height):  
        for x in range(width):
            # get the direction.  
            # camera points in -x direction, FOV = 90 degrees  
            x_render, y_render = (x-int(width/2))/width, (y-int(height/2))/height
            dir = mathutils.Vector((x_render, y_render, -1))
            dir.rotate(rotation)
             
            # cast a ray into the scene  
            hit, loc, normal, index, ob, mat = depsgraph.scene.ray_cast(depsgraph, origin, dir)

            if hit:
                if "isBH" in ob.data:
                    buf[y,x,0:3] = blackhole_hit(SW, approx, depsgraph, dir, background, object_tex, buf, lamps, hit, loc, normal, index, ob, mat)
                else:
                    buf[y,x,0:3] = normal_hit(depsgraph, object_tex, lamps, hit, loc, normal, index, ob, mat)
            else:
                buf[y,x,0:3] = background_hit(dir, background)
    return buf


# straight from https://docs.blender.org/api/current/bpy.types.RenderEngine.html?highlight=renderengine
class CustomRenderEngine(bpy.types.RenderEngine):
    bl_idname = "gr_ray_tracer"
    bl_label = "General Relativistic Renderer"
    bl_use_preview = True


    image_name_bg = "8k_stars_milky_way.jpg"# 
#    image_name = "Untitled"# #"bg_orion.png"
    tex_name_bg = image_name_bg #"MP2"
    
    image_name_stars = "8k_moon.jpg"#"8k_sun.jpg"
    tex_name_stars = image_name_stars
    
    def render(self, depsgraph):
        
        loadTexture(self.image_name_bg, self.tex_name_bg)
        loadTexture(self.image_name_stars, self.tex_name_stars)
        
        if self.is_preview:  # we might differentiate later
            pass             # for now ignore completely
        else:
            self.render_scene(depsgraph, self.tex_name_bg, self.tex_name_stars)

    def render_scene(self, depsgraph, background_tex, object_tex):
        scale = depsgraph.scene.render.resolution_percentage / 100.0
        res_x = int(depsgraph.scene.render.resolution_x*scale)
        res_y = int(depsgraph.scene.render.resolution_y*scale)

        buf = ray_trace(depsgraph, res_x, res_y, background_tex, object_tex)
        buf.shape = -1,4

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, res_x, res_y)#self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = buf.tolist()
        self.end_result(result)




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


class EXAMPLE_PT_panel_1(bpy.types.Panel):
    bl_label = "Panel 1"
    bl_category = "Example tab"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "world"
    bl_options = {"DEFAULT_CLOSED"}
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="This is panel 1.")
        
        row = layout.row()
        row.label(text="Hello world!", icon='WORLD_DATA')

        for (prop_name, _) in PROPS:
            row = layout.row()
            row.prop(context.scene, prop_name)
        
        

def register():
    # Register the RenderEngine
    bpy.utils.register_class(CustomRenderEngine)
    bpy.utils.register_class(EXAMPLE_PT_panel_1)

    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)


    for panel in get_panels():
        panel.COMPAT_ENGINES.add('gr_ray_tracer')#CUSTOM')
    
#    <class 'cycles.ui.NODE_CYCLES_WORLD_PT_ray_visibility'>
#<class 'cycles.ui.NODE_CYCLES_WORLD_PT_settings'>
#<class 'cycles.ui.NODE_CYCLES_WORLD_PT_settings_surface'>
#<class 'cycles.ui.NODE_CYCLES_WORLD_PT_settings_volume'>
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
    bpy.utils.unregister_class(EXAMPLE_PT_panel_1)
    
    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)
        
        
    for panel in get_panels():
        if 'gr_ray_tracer' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('gr_ray_tracer')#CUSTOM')

if __name__ == "__main__":
    register()
