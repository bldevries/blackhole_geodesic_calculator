


# (base) vries001@ bin % pwd
# /Applications/Blender_versions/Blender4.1.app/Contents/Resources/4.1/python/bin
# (base) vries001@ bin % ./python3.11 -m pip install ~/Dropbox/0_DATA_BEN/PHYSICS/PROJECTS/blackhole_geodesic_calculator/python_pkg_dev/ -U


# Needed for addon:
# bl_info = {
#     "name": "general_relativistic_ray_tracer",
#     "author": "B.L. de Vries",
#     "version": (0, 0, 202407100),
#     "blender": (4, 1, 0),
#     "location": "",
#     "description": "",
#     "warning": "",
#     "wiki_url": "",
#     "category": "Render",
# }

# For when I want to fix and add panels and options:

# def get_panels():
#     exclude_panels = {
#         'VIEWLAYER_PT_filter',
#         'VIEWLAYER_PT_layer_passes',
#     }

#     panels = []
#     for panel in bpy.types.Panel.__subclasses__():
#         #print(panel)
#         if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
#             if panel.__name__ not in exclude_panels:
#                 panels.append(panel)

#     return panels

# PROPS = [
#     ('background_image_renderer', bpy.props.StringProperty(name='File name', default='...', subtype='FILE_NAME')),

# ]


# class EXAMPLE_PT_panel_1(bpy.types.Panel):
#     bl_label = "Panel 1"
#     bl_category = "Example tab"
#     bl_space_type = "PROPERTIES"
#     bl_region_type = "WINDOW"
#     bl_context = "world"
#     bl_options = {"DEFAULT_CLOSED"}
    
#     def draw(self, context):
#         layout = self.layout
#         layout.label(text="This is panel 1.")
        
#         row = layout.row()
#         row.label(text="Hello world!", icon='WORLD_DATA')

#         for (prop_name, _) in PROPS:
#             row = layout.row()
#             row.prop(context.scene, prop_name)
        