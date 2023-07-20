import math
import numpy as np
import matplotlib.pylab as plt
import os

from pyparsing import indentedBlock

from classes.drone import Drone
from classes.payload import Payload
from util import mujoco_helper


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

            itl = si + offset_x1
            jtl = sj + offset_y

            itr = si - offset_x2
            jtr = sj + offset_y

            ibl = si + offset_x1
            jbl = sj - offset_y

            ibr = si - offset_x2
            jbr = sj - offset_y

            if(itl < slice_upscaled.shape[0] and jtl < slice_upscaled.shape[1]):
                vtl = slice_upscaled_mirrored[itl, jtl]
                #vtl = slice_upscaled[itl, jtl]

            if(itr >= 0 and jtr < slice_upscaled.shape[1]):
                vtr = slice_upscaled[itr, jtr]

            if(ibl < slice_upscaled.shape[0] and jbl >= 0):
                vbl = slice_upscaled[ibl, jbl]
                
            if(ibr >= 0 and jbr >= 0):
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


abs_path = os.path.dirname(os.path.abspath(__file__))
data_file_name = os.path.join(abs_path, "..", "airflow_data", "raw_airflow_data", "dynamic_pressure_field_processed_new.csv")
tmp = np.loadtxt(mujoco_helper.skipper(data_file_name), delimiter=',', dtype=np.float64)
# transform data into 3D array
cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
data = np.reshape(tmp[:, 4], (cube_size, cube_size, cube_size))
print(data.shape)


combined_data = np.empty_like(data)

for i in range(cube_size):
    print("computing slice: " + str(i))
    slice_shifted = create_shifted_slice(data[:, :, i], 100, 65, 87)

    combined_data[:, :, i] = slice_shifted
combined_1d = combined_data.reshape(-1)
np.savetxt(os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_pressure_shifted.txt"), combined_1d)

#tmp = np.loadtxt(os.path.join(abs_path, "..", "airflow_data", "airflow_luts", "flow_pressure_shifted.txt"))
#cube_size = int(math.pow(tmp.shape[0] + 1, 1/3))
#combined_data = np.reshape(tmp, (cube_size, cube_size, cube_size))
#print(combined_data.shape)

#plt.imshow(data[:, :, 0], cmap='jet', interpolation='nearest')
plt.imshow(combined_data[:, :, 40], cmap='jet', interpolation='nearest')
plt.show()