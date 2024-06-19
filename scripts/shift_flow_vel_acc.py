import math
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from aiml_virtual.object.drone import BUMBLEBEE_PROP

from pyparsing import indentedBlock
from sympy import intervals

from aiml_virtual.util import mujoco_helper

from matplotlib.animation import FuncAnimation 


def create_shifted_slice(slice_, offset_x1, offset_x2, offset_y):
    cube_size = slice_.shape[0]
    scale = 10
    slice_upscaled = np.empty((cube_size * scale, cube_size * scale, 3))

    i = 0
    for si in range(cube_size * scale):
        j = 0
        for sj in range(cube_size * scale):
            slice_upscaled[si, sj] = slice_[i, j]
            if((sj + 1) % scale == 0):
                j += 1 
        
        if((si + 1) % scale == 0):
            i += 1
    
    #slice_upscaled_mirrored = np.flipud(slice_upscaled_mirrored)
    slice_upscaled_mirrored = np.fliplr(np.copy(slice_upscaled))

    #slice_upscaled_mirrored[:, :, 0] = -slice_upscaled_mirrored[:, :, 0]
    slice_upscaled_mirrored[:, :, 1] = -slice_upscaled_mirrored[:, :, 1]

    slice_upscaled_shifted = np.empty_like(slice_upscaled)


    for si in range(cube_size * scale):
        for sj in range(cube_size * scale):
            vtl = vtr = vbl = vbr = 0

            itr = si + offset_x2 # prop4 in xml
            jtr = sj + offset_y

            itl = si - offset_x1 # prop3 in xml
            jtl = sj + offset_y

            ibr = si + offset_x2 # prop1 in xml 
            jbr = sj - offset_y

            ibl = si - offset_x1 # prop2 in xml
            jbl = sj - offset_y

            if(itl >= 0 and jtl < slice_upscaled.shape[1]):
                vtl = slice_upscaled_mirrored[itl, jtl]
                #vtl = slice_upscaled[itl, jtl]

            if(itr < slice_upscaled.shape[0] and jtr < slice_upscaled.shape[1]):
                vtr = slice_upscaled[itr, jtr]

            if(ibl >= 0 and jbl >= 0):
                vbl = slice_upscaled[ibl, jbl]
                
            if(ibr < slice_upscaled.shape[0] and jbr >= 0):
                vbr = slice_upscaled_mirrored[ibr, jbr]
                #vbr = slice_upscaled[ibr, jbr]
            
            val = vtl + vtr + vbl + vbr
            slice_upscaled_shifted[si, sj] = val


    slice_shifted = np.empty_like(slice_)

    for i in range(cube_size):
        for j in range(cube_size):
            sumv = 0
            for si in range(scale):
                for sj in range(scale):
                    sumv += slice_upscaled_shifted[i * scale + si, j * scale + sj]
            
            slice_shifted[i, j] = sumv / (scale * scale)
    
    return slice_shifted

offset_y = int(round(float(BUMBLEBEE_PROP.OFFSET_Y.value) * 1000)) # convert to mm
offset_x1 = int(round(float(BUMBLEBEE_PROP.OFFSET_X1.value) * 1000))
offset_x2 = int(round(float(BUMBLEBEE_PROP.OFFSET_X2.value) * 1000))

SLICE = 40

abs_path = os.path.dirname(os.path.abspath(__file__))
data_file_name = os.path.join(abs_path, "..", "airflow_data", "raw_airflow_data", "single_rotor_velocity.csv")
tmp = np.loadtxt(mujoco_helper.skipper(data_file_name), delimiter=',', dtype=np.float64)
# transform data into 3D array
cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
data = np.reshape(tmp[:, 3], (cube_size, cube_size, cube_size))

velocities_xyz_normalized = np.empty_like(tmp[:, 3:])
for i in range(len(velocities_xyz_normalized)):
    v = tmp[:, 3:][i]
    #l = tmp[:, 3][i]
    l = np.sqrt(tmp[:, 3][i]**2 + tmp[:, 4][i]**2 + tmp[:, 5][i]**2)
    velocities_xyz_normalized[i] = v
    if np.abs(l) > 1e-3:
        velocities_xyz_normalized[i] /= l

velocities_xyz_normalized = np.reshape(velocities_xyz_normalized, (cube_size, cube_size, cube_size, 3))
tail_xyz = np.reshape(tmp[:, :3], (cube_size, cube_size, cube_size, 3)) * 100 # convert to cm
print(velocities_xyz_normalized.shape)


#plt.imshow(data[:, :, SLICE], cmap='jet', interpolation='nearest')
#plt.show()



# --------------------------------- plot original slice --------------------------------------

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#
#x = tail_xyz[:, :, SLICE, 0]
#y = tail_xyz[:, :, SLICE, 1]
#z = tail_xyz[:, :, SLICE, 2]
#
##x, y, z = np.meshgrid(np.arange(0, 50, 1), np.arange(0, 50, 1), np.zeros(50))
#
#u = velocities_xyz_normalized[:, :, SLICE, 0]
#v = velocities_xyz_normalized[:, :, SLICE, 1]
#w = velocities_xyz_normalized[:, :, SLICE, 2]
#
#colormap = cm.jet
#
#lens_slice = np.reshape(data[:, :, SLICE], (cube_size * cube_size))
#print(np.max(lens_slice))
#norm = Normalize()
#norm.autoscale(lens_slice)
#
#temp_cm = colormap(norm(lens_slice)).tolist()
#
##heatmap = np.vstack((heatmap, heatmap, heatmap))
#heatmap = temp_cm[:]
#
#for h in temp_cm:
#    heatmap.append(h)
#    heatmap.append(h)
#
#
#ax.quiver(x, y, z, u, v, w, color=heatmap)
##ax.quiver(x[:, :, 0], y[:, :, 0], z[:, :, 0], u, v, w, c=lenghts[:, :, SLICE])
#
## make the panes transparent
#ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
## make the grid lines transparent
#ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#ax.axis("equal")
#
#plt.show()


# --------------------------------- plot shifted slice --------------------------------------

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#
velocities_xyz = np.reshape(tmp[:, 3:], (cube_size, cube_size, cube_size, 3))
#
#x = tail_xyz[:, :, SLICE, 0]
#y = tail_xyz[:, :, SLICE, 1]
#z = tail_xyz[:, :, SLICE, 2]
#
##x, y, z = np.meshgrid(np.arange(0, 50, 1), np.arange(0, 50, 1), np.zeros(50))
#
#slice_shifted = create_shifted_slice(velocities_xyz[:, :, SLICE, :], 100, 65, 87)
#lengths_shifted = np.empty((cube_size * cube_size))
#i = 0
#for row in slice_shifted:
#    for v in row:
#        #print(v)
#        l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
#        v /= l
#        lengths_shifted[i] = l
#        i += 1
#
#print(np.max(lengths_shifted))
#
#u = slice_shifted[:, :, 0]
#v = slice_shifted[:, :, 1]
#w = slice_shifted[:, :, 2]
#
#colormap = cm.jet
#
##lens_slice = np.reshape(data[:, :, SLICE], (cube_size * cube_size))
#norm = Normalize()
#norm.autoscale(lengths_shifted)
#
#temp_cm = colormap(norm(lengths_shifted)).tolist()
#
#heatmap = temp_cm[:]
#
#for h in temp_cm:
#    heatmap.append(h)
#    heatmap.append(h)
#
#
#ax.quiver(x, y, z, u, v, w, color=heatmap)
##ax.quiver(x[:, :, 0], y[:, :, 0], z[:, :, 0], u, v, w, c=lenghts[:, :, SLICE])
#
## make the panes transparent
#ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
## make the grid lines transparent
#ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#ax.axis("equal")
#
#plt.show()


# ---------------------------------------- animation ----------------------------------------------

slices_shifted_not_normalized = np.empty((cube_size, cube_size, cube_size, 3))
slices_shifted = []
heatmaps = []

for i in range(cube_size):
    print("computing slice " + str(i))
    slice_shifted = create_shifted_slice(velocities_xyz[:, :, i, :], offset_x1, offset_x2, offset_y)
    slices_shifted_not_normalized[:, :, i, :] = np.copy(slice_shifted)
    lengths_shifted = np.empty((cube_size * cube_size))
    i = 0
    for row in slice_shifted:
        for v in row:
            #print(v)
            l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            v /= l
            lengths_shifted[i] = l
            i += 1
    
    colormap = cm.jet

    #lens_slice = np.reshape(data[:, :, SLICE], (cube_size * cube_size))
    norm = Normalize()
    norm.autoscale(lengths_shifted)

    temp_cm = colormap(norm(lengths_shifted)).tolist()

    #heatmap = np.vstack((heatmap, heatmap, heatmap))
    heatmap = temp_cm[:]

    for h in temp_cm:
        heatmap.append(h)
        heatmap.append(h)
    
    slices_shifted += [slice_shifted]
    heatmaps += [heatmap]


slices_shifted_not_normalized = np.array(slices_shifted_not_normalized)

slices_shifted_not_normalized = slices_shifted_not_normalized.reshape((cube_size**3, 3))
np.savetxt(os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "openfoam_velocity.txt"), slices_shifted_not_normalized)
