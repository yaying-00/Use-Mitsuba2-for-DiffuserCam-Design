import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Bitmap, Struct, Thread, xml, UInt32, Float, Vector2f, Vector3f, Transform4f, ScalarTransform4f
from mitsuba.core.xml import load_string
from mitsuba.render import SurfaceInteraction3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam, SGD
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from scipy.ndimage.filters import gaussian_filter

# Change tasks to complete different optimization
task = 'plain2bumpy'
# task = 'bumpy2plain'
# task = 'bumpy2bumpy'

# The size of gaussian used for smoothing between optimizations
smooth_sigma = 0.7

# Convert flat array into a vector of arrays (will be included in next enoki release)
def ravel(buf, dim = 3):
	idx = dim * UInt32.arange(ek.slices(buf) // dim)
	if dim == 2:
		return Vector2f(ek.gather(buf, idx), ek.gather(buf, idx + 1))
	elif dim == 3:
		return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim = 3):
	idx = UInt32.arange(ek.slices(source))
	for i in range(dim):
		ek.scatter(target, source[i], dim * idx + i)

def plot_contour(para_raveled, title, sigma = 0, plot_surf = True):
	vertex_pos = para_raveled
	vertex_np = np.array(vertex_pos)
	vertex_np = vertex_np[vertex_np[:,1].argsort()]
	vertex_np = vertex_np[vertex_np[:,0].argsort(kind='mergesort')]
	X = np.reshape(vertex_np[:,0], (256, 256)).T
	Y = np.reshape(vertex_np[:,1], (256, 256)).T
	Z = np.reshape(vertex_np[:,2], (256, 256)).T

	if plot_surf:
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
		print('Writing ' + output_path + title + '.png')
		plt.savefig(output_path + title + '.png')

	fig = plt.figure()
	cp = plt.contourf(X, Y, Z)
	plt.colorbar(cp)
	print('Writing ' + output_path + title + '_contour.png')
	plt.savefig(output_path + title + '_contour.png')


	if sigma != 0:
		Z_smooth = gaussian_filter(Z, sigma = sigma)

		if plot_surf:
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			ax.plot_surface(X, Y, Z_smooth, cmap='viridis', edgecolor='none')
			print('Writing ' + output_path + title + '_smooth' + '.png')
			plt.savefig(output_path + title + '_smooth' + '.png')

		fig = plt.figure()
		cp = plt.contourf(X, Y, Z_smooth)
		plt.colorbar(cp)
		print('Writing ' + output_path + title + '_contour' + '_smooth' + '.png')
		plt.savefig(output_path + title + '_contour' + '_smooth' + '.png')

def make_scene(integrator, spp, w=256, h=256):
	return load_string(
	"""
	<?xml version="1.0" encoding="utf-8"?>
	<scene version="2.1.0">
		{integrator}

		<!-- Museum environment map is turned off-->
		<!--
		<emitter type="envmap" id="my_envmap">
			<string name="filename" value="scene/museum.exr"/>
		</emitter>
		-->

		<!-- Perspective pinhole camera -->
		<sensor type="perspective">
			<transform name="to_world">
				<lookat origin="0.0, 0.0, 3.0"
						target="0.0, 0.0, 0.0"
						up="0.0, 1.0, 0.0"/>
			</transform>

			<float name="fov" value="40"/>

			<film type="hdrfilm">
				<string name="pixel_format" value="rgb"/>
				<integer name="width" value="{w}"/>
				<integer name="height" value="{h}"/>
				<rfilter type="box"/>
			</film>

			<sampler type="independent">
				<integer name="sample_count" value="{spp}"/>
			</sampler>
		</sensor>

		<!-- spherical emitter -->
		<shape type="sphere" id="sphere_emitter">
			<emitter type="area">
				<spectrum name="radiance" value="10.0"/>
			</emitter>
			<point name="center" x="0" y="0" z="-100"/>
			<float name="radius" value="3"/>
		</shape>

		<!-- Can change the spherical emitter to a rectangular one --> 
		<!--
		<shape type="rectangle" id="lightsource">
			<transform name="to_world">
				<scale value="2"/>
				<rotate x="1" angle="180"/>
				<translate x="0.0" y="0.0" z="5"/>
			</transform>

			<emitter type="smootharea">
				<rgb name="radiance" value="5.0"/>
			</emitter>
		</shape>
		-->

		<!-- Scattering model of the glass panel -->
		<bsdf type="roughdielectric" id="face">
			<float name="int_ior" value="1.51"/>
			<float name="ext_ior" value="1.0"/>
			<string name="distribution" value="beckmann"/>
			<float name="alpha" value="0.01"/>
		</bsdf>

		<!-- The diffuser surface to optimize -->
		<shape type="ply" id="grid_mesh">
			<string name="filename" value="scene/grid_256.ply"/>
			<transform name="to_world">
				<scale value="1"/>
				<translate x="0" y="0" z="0"/>
			</transform>
			<ref id = "face"/>
		</shape>

		<!-- Other surfaces of the diffuser -->
		<shape type="obj">
			<string name="filename" value="scene/glass_noback_1.obj"/>
			<ref id = "face"/>
		</shape>

		<!-- Scattering model of the aperture -->
		<bsdf type="diffuse" id="aperture">
			<rgb name="reflectance" value="0.0, 0.0, 0.0"/>
		</bsdf>
		
		<!-- Aperture around the diffuser -->
		<shape type="obj">
			<string name="filename" value="scene/aperture.obj"/>
			<ref id = "aperture"/>
		</shape>
			
		</scene>
	""".format(integrator=integrator, spp=spp, w=w, h=h)
	)

path_str =  """
<integrator type="path">
	<integer name="max_depth" value="4"/>
</integrator>"""

path_reparam_str =  """
<integrator type="pathreparam">
	<integer name="max_depth" value="4"/>
	<boolean name="use_variance_reduction" value="true"/>
	<boolean name="use_convolution" value="true"/>
	<boolean name="disable_gradient_diffuse" value="false"/>
</integrator>"""

# width and height determines the resolution of rendered image
width   = 128
height  = 128
# spp: samples per pixel
spp_ref = 64
spp_opt = 8

# Prepare output folder
output_path = 'output/diffuser_smooth/' + task + '/'

if not os.path.isdir(output_path):
	os.makedirs(output_path)

# Generate the scene and image with a plain glass panel
scene = make_scene(path_str, spp_ref, width, height)
image_plain = render(scene)
write_bitmap(output_path + "out_plain.png", image_plain, (width, height))
print("Writing " + "out_plain.png")

params = traverse(scene)
print(params)
positions_buf = params['grid_mesh.vertex_positions_buf']
positions_initial = ravel(positions_buf)
normals_initial = ravel(params['grid_mesh.vertex_normals_buf'])
vertex_count = ek.slices(positions_initial)

filename = 'scene/diffuser_surface_1.jpg'
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Create a texture with the reference displacement map
disp_tex_1 = xml.load_dict({
	"type" : "bitmap",
	"filename": "diffuser_surface_1.jpg",
	"to_uv" : ScalarTransform4f.scale([1, -1, 1]) # texture is upside-down
}).expand()[0]

# Create a texture with another displacement map
disp_tex_2 = xml.load_dict({
        "type" : "bitmap",
        "filename": "diffuser_surface_2.jpg",
        "to_uv" : ScalarTransform4f.scale([1, -1, 1]) # texture is upside-down
}).expand()[0]

# Create a fake surface interaction with an entry per vertex on the mesh
mesh_si = SurfaceInteraction3f.zero(vertex_count)
mesh_si.uv = ravel(params['grid_mesh.vertex_texcoords_buf'], dim=2)

# Evaluate the displacement map for the entire mesh
disp_tex_diffuser_1 = disp_tex_1.eval_1(mesh_si)
disp_tex_diffuser_2 = disp_tex_2.eval_1(mesh_si)

# Apply displacement to mesh vertex positions and update scene (e.g. OptiX BVH)
def apply_displacement(disp, amplitude = 0.05):
	new_positions = disp.eval_1(mesh_si) * normals_initial * amplitude + positions_initial
	unravel(new_positions, params['grid_mesh.vertex_positions_buf'])
	params.set_dirty('grid_mesh.vertex_positions_buf')
	params.update()

# Change amp to adjust the magnitude of displacement
amp = 0.1

# Apply displacement before generating reference image
apply_displacement(disp_tex_1, amp)

# Render a reference image (no derivatives used yet)
image_ref = render(scene, spp=32)
print('Write ' + output_path + 'out_ref.png')
write_bitmap(output_path + 'out_ref.png', image_ref, (width, height))

# Plot the reference height map of the diffuser
plot_contour(ravel(params['grid_mesh.vertex_positions_buf']), title = 'diffuer_ref')

# Create another scene for optimizing geometry parameters
del scene
scene = make_scene(path_reparam_str, spp_opt, width, height)

vertex_pos_key = 'grid_mesh.vertex_positions_buf'
params = traverse(scene)
params.keep([vertex_pos_key])
print('Parameter map after filtering: ', params)

vertex_positions_buf = params[vertex_pos_key]
vertex_positions = ravel(vertex_positions_buf)
vertex_count = ek.slices(vertex_positions)

if task == 'plain2bumpy':
	disp_tex_data_init = ek.full(Float, 0.0, vertex_count)
	disp_tex_data_ref = disp_tex_diffuser_1
	displacements = ek.full(Float, 0.0, vertex_count)

if task == 'bumpy2plain':
	disp_tex_data_init = disp_tex_diffuser_1
	disp_tex_data_ref = ek.full(Float, 0.0, vertex_count)
	displacements = disp_tex_diffuser_1 * amp

if task == 'bumpy2bumpy':
	disp_tex_data_init = disp_tex_diffuser_2
	disp_tex_data_ref = disp_tex_diffuser_1
	displacements = disp_tex_diffuser_2 * amp

params_opt = {"displacements": displacements}
if task == 'bumpy2bumpy':
	plot_contour(vertex_positions + Vector3f(0,0,1) * params_opt['displacements'], 'diffuser_init', sigma = 0, plot_surf = False)

loss_list = []
diff_render_init = []
diff_vertex_ref = []
diff_vertex_init = []

for j in range(10):
	opt = Adam(params_opt, lr = 0.001)
	# opt = SGD(params_opt, lr = 0.001, momentum = 0.9)
	for i in range(10):
		unravel(vertex_positions + Vector3f(0,0,1) * params_opt['displacements'], params[vertex_pos_key])
		params.set_dirty(vertex_pos_key)
		params.update()
		
		image = render(scene)
		if j == 0 and i == 0:
			image_init = image
		
		if ek.any(ek.any(ek.isnan(params[vertex_pos_key]))):
			print("[WARNING] NaNs in the vertex positions.")
		
		if ek.any(ek.isnan(image)):
			print("[WARNING] NaNs in the image.")
			
		# Write a gamma encoded PNG
		image_np = image.numpy().reshape(height, width, 3)
		output_file = output_path + 'out_%03i_%03i.png' % (j, i)
		print("Writing image %s" % (output_file))
		Bitmap(image_np).convert(pixel_format=Bitmap.PixelFormat.RGB, component_format=Struct.Type.UInt8, srgb_gamma=True).write(output_file)
		
		# Objective function
		if task == 'plain2bumpy' or task == 'bumpy2bumpy':
			loss = ek.hsum(ek.hsum(ek.sqr(image - image_ref))) / (height*width*3)
		if task == 'bumpy2plain':
			loss = ek.hsum(ek.hsum(ek.sqr(image - image_plain))) / (height*width*3)

		print("Iteration %i-%i: loss=%f" % (j, i, loss[0]))
		loss_list.append(loss[0])

		diff_image_init  = ek.hsum(ek.hsum(ek.sqr(image - image_init))) / (height*width*3)
		#print("difference with initial image = %f" % (diff_image_init[0]))
		diff_render_init.append(diff_image_init[0])

		diff_init = ek.hsum(ek.sqr(params_opt['displacements'] - disp_tex_data_init * amp)) / ek.slices(params_opt['displacements'])
		diff_ref = ek.hsum(ek.sqr(params_opt['displacements'] - disp_tex_data_ref * amp)) / ek.slices(params_opt['displacements'])
		
		#print("diff_init:", diff_init[0])
		#print("diff_ref:", diff_ref[0])
		diff_vertex_init.append(diff_init[0])
		diff_vertex_ref.append(diff_ref[0])
		

		if(loss[0] != loss[0]):
			print("[WARNING] Skipping current iteration due to NaN loss.")
			continue

		ek.backward(loss)

		if ek.any(ek.isnan(ek.gradient(params_opt['displacements']))):
				print("[WARNING] NaNs in the displacement gradients. ({iteration:d})".format(iteration=i))
				exit(-1)

		opt.step()

		if ek.any(ek.isnan(params_opt['displacements'])):
			print("[WARNING] NaNs in the vertex displacements. ({iteration:d})".format(iteration=i))
			exit(-1)

	vertex_pos_optim = ravel(params[vertex_pos_key])
	vertex_np = np.array(vertex_pos_optim)
	ind = np.linspace(0, vertex_np.shape[0]-1, num=vertex_np.shape[0])
	ind = np.reshape(ind,(-1,1))
	vertex_np = np.concatenate((vertex_np, ind), axis = 1)
	vertex_np = vertex_np[vertex_np[:,1].argsort()]
	vertex_np = vertex_np[vertex_np[:,0].argsort(kind='mergesort')]
	Z = np.reshape(vertex_np[:,2], (256, 256)).T
	Z_smooth = gaussian_filter(Z, sigma = smooth_sigma)
	vertex_np[:,2] =  np.reshape(Z_smooth, (-1,))
	vertex_np = vertex_np[vertex_np[:,3].argsort()]
	vertex_np = vertex_np[:, 0:3]
	params_opt['displacements'] = Float(vertex_np[:,2])

# Plot the loss curve
fig, ax = plt.subplots()
ax.plot(loss_list, '-b', label = 'Squared error between image_render and image_ref')
ax.plot(diff_render_init, '-r', label = 'Squared error between image_render and image_init')
leg = ax.legend()
plt.title('loss')
plt.xlabel('iteration')
plt.savefig(output_path + 'loss.png')

# Plot the difference between the vertex positions
fig, ax = plt.subplots()
ax.plot(diff_vertex_ref, '-b', label = 'Squared error between vertex_render and vertex_ref')
ax.plot(diff_vertex_init, '-r', label = 'Squared error between vertex_render and vertex_init')
leg = ax.legend()
plt.title('vertex position difference')
plt.xlabel('iteration')
plt.savefig(output_path + 'vertex_position.png')

# Plot the diffuser height map after optimization
plot_contour(ravel(params[vertex_pos_key]), title = 'diffuer_optm', sigma = 1.0)
print("DONE.")




