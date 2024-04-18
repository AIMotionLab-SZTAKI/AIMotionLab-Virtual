from cProfile import label
import math
import numpy as np
import matplotlib.pylab as plt
import os
from aiml_virtual.util import mujoco_helper
from aiml_virtual.object.drone import BUMBLEBEE_PROP



def create_shifted_slice(slice_, offset_x1, offset_x2, offset_y):
    cube_size = slice_.shape[0]
    scale = 10
    slice_upscaled = np.empty((cube_size * scale, cube_size * scale))

    i = 0
    for si in range(cube_size * scale):
        j = 0
        for sj in range(cube_size * scale):
            slice_upscaled[si, sj] = slice_[i, j]
            if((sj + 1) % scale == 0):
                j += 1 
        
        if((si + 1) % scale == 0):
            i += 1

    slice_upscaled_mirrored = np.fliplr(slice_upscaled)
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


USE_EXISTING_DATA = True


abs_path = os.path.dirname(os.path.abspath(__file__))


if USE_EXISTING_DATA:

    tmp = np.loadtxt(os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_pressure_shifted.txt"))
    cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
    combined_data = np.reshape(tmp, (cube_size, cube_size, cube_size))
    print(combined_data.shape)

else:
    data_file_name = os.path.join(abs_path, "..", "airflow_data", "raw_airflow_data", "dynamic_pressure_field_processed_new.csv")
    tmp = np.loadtxt(mujoco_helper.skipper(data_file_name), delimiter=',', dtype=np.float64)
    # transform data into 3D array
    cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
    data = np.reshape(tmp[:, 4], (cube_size, cube_size, cube_size))
    print(data.shape)
    
    combined_data = np.empty_like(data)
    
    offset_y = int(round(float(BUMBLEBEE_PROP.OFFSET_Y.value) * 1000)) # convert to mm
    offset_x1 = int(round(float(BUMBLEBEE_PROP.OFFSET_X1.value) * 1000))
    offset_x2 = int(round(float(BUMBLEBEE_PROP.OFFSET_X2.value) * 1000))
    
    for i in range(cube_size):
        print("computing slice: " + str(i))
        slice_shifted = create_shifted_slice(data[:, :, i], offset_x1, offset_x2, offset_y)
    
        combined_data[:, :, i] = slice_shifted
    combined_1d = combined_data.reshape(-1)
    np.savetxt(os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_pressure_shifted.txt"), combined_1d)

#plt.imshow(data[:, :, 0], cmap='jet', interpolation='nearest')
#im = plt.imshow(np.rot90(combined_data[15, :, :]), cmap='jet', interpolation='nearest')
#colorbar = plt.colorbar(im, label="Pressure [Pa]")
#plt.show()

# create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# plot the pressure heatmap for the xy plane on the first subplot
#z_value = cube_size - 1
z_value = 0
pressure_map_z = np.copy(combined_data[:, :, z_value])
#pressure_map_z[40, 30] = 500
im1 = axs[0].imshow(pressure_map_z.T, cmap='jet', interpolation='nearest')
axs[0].set_xlabel('x [cm]')
axs[0].set_ylabel('y [cm]')
axs[0].invert_yaxis()
axs[0].set_title(f'Pressure at z={z_value} [cm]')
fig.colorbar(im1, ax=axs[0], label='Pressure [Pa]')

# plot the pressure heatmap for the yz plane on the second subplot
x_value = 15

pressure_map_x = np.copy(combined_data[x_value, :, :])
#pressure_map_x[40, 30] = 500
im2 = axs[1].imshow(pressure_map_x.T, cmap='jet', interpolation='nearest')
axs[1].set_xlabel('y [cm]')
axs[1].set_ylabel('z [cm]')
axs[1].invert_yaxis()
axs[1].set_title(f'Pressure at x={x_value} [cm]')
fig.colorbar(im2, ax=axs[1], label='Pressure [Pa]')
# set aspect ratio to equal for both subplots
#axs[0].set_aspect('equal', 'box')
#axs[1].set_aspect('equal', 'box')

# adjust the layout of the subplots to prevent overlap
fig.tight_layout()

plt.show()