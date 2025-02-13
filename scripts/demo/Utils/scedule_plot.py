import matplotlib.pyplot as plt
import numpy as np
"""
---------------------------------0.02----------------------------------------
Drone 0: Go from v0 = [-4.  -4.5  2. ] to v19 = [-4.5 -3.   2. ]
Suces_rte =  [1, 1]
Drone 0: A* done under 0.0009999275207519531 sec
Drone 0: Trajectory start: 2.02 sec
Drone 0: Trajectory end: 4.82 sec
Drone 0: Trajectory duration: 2.8000000000000003 sec
---------------------------------0.04----------------------------------------
Drone 1: Go from v1 = [-3.  -4.5  2. ] to v20 = [-4.5 -2.   2. ]
Suces_rte =  [1, 1]
Drone 1: A* done under 0.020004749298095703 sec
Drone 1: Trajectory start: 2.04 sec
Drone 1: Trajectory end: 8.24 sec
Drone 1: Trajectory duration: 6.2 sec
---------------------------------0.08----------------------------------------
Drone 2: Go from v2 = [-2.  -4.5  2. ] to v28 = [ 4.5 -3.   2. ]
Suces_rte =  [1, 1]
Drone 2: A* done under 0.017999649047851562 sec
Drone 2: Trajectory start: 2.08 sec
Drone 2: Trajectory end: 13.48 sec
Drone 2: Trajectory duration: 11.4 sec
---------------------------------0.1----------------------------------------
Drone 3: Go from v3 = [-1.  -4.5  2. ] to v2 = [-2.  -4.5  2. ]
Suces_rte =  [1, 1]
Drone 3: A* done under 0.0019998550415039062 sec
Drone 3: Trajectory start: 2.1 sec
Drone 3: Trajectory end: 6.1 sec
Drone 3: Trajectory duration: 3.9999999999999996 sec
---------------------------------0.12----------------------------------------
Drone 4: Go from v4 = [ 0.  -4.5  2. ] to v1 = [-3.  -4.5  2. ]
Suces_rte =  [1, 1]
Drone 4: A* done under 0.1230006217956543 sec
Drone 4: Trajectory start: 2.12 sec
Drone 4: Trajectory end: 12.92 sec
Drone 4: Trajectory duration: 10.8 sec
---------------------------------0.26----------------------------------------
Drone 5: Go from v5 = [ 1.  -4.5  2. ] to v34 = [4.5 3.  2. ]
Suces_rte =  [1, 1]
Drone 5: A* done under 0.011005163192749023 sec
Drone 5: Trajectory start: 2.26 sec
Drone 5: Trajectory end: 14.56 sec
Drone 5: Trajectory duration: 12.3 sec
---------------------------------0.28----------------------------------------
Drone 6: Go from v6 = [ 2.  -4.5  2. ] to v23 = [-4.5  1.   2. ]
Suces_rte =  [1, 1]
Drone 6: A* done under 0.024000167846679688 sec
Drone 6: Trajectory start: 2.28 sec
Drone 6: Trajectory end: 19.28 sec
Drone 6: Trajectory duration: 17.0 sec
---------------------------------0.32----------------------------------------
Drone 7: Go from v7 = [ 3.  -4.5  2. ] to v16 = [3.  4.5 2. ]
Suces_rte =  [1, 1]
Drone 7: A* done under 0.009459972381591797 sec
Drone 7: Trajectory start: 2.32 sec
Drone 7: Trajectory end: 14.52 sec
Drone 7: Trajectory duration: 12.2 sec
---------------------------------0.34----------------------------------------
Drone 8: Go from v8 = [ 4.  -4.5  2. ] to v3 = [-1.  -4.5  2. ]
Suces_rte =  [1, 1]
Drone 8: A* done under 0.006999969482421875 sec
Drone 8: Trajectory start: 2.34 sec
Drone 8: Trajectory end: 8.74 sec
Drone 8: Trajectory duration: 6.4 sec
---------------------------------0.36----------------------------------------
Drone 9: Go from v9 = [-4.   4.5  2. ] to v17 = [4.  4.5 2. ]
Suces_rte =  [1, 1]
Drone 9: A* done under 0.03999972343444824 sec
Drone 9: Trajectory start: 2.36 sec
Drone 9: Trajectory end: 17.06 sec
Drone 9: Trajectory duration: 14.7 sec
---------------------------------0.4----------------------------------------
Drone 10: Go from v10 = [-3.   4.5  2. ] to v8 = [ 4.  -4.5  2. ]
Suces_rte =  [1, 1]
Drone 10: A* done under 0.02603602409362793 sec
Drone 10: Trajectory start: 2.4 sec
Drone 10: Trajectory end: 18.4 sec
Drone 10: Trajectory duration: 15.999999999999998 sec
---------------------------------0.44----------------------------------------
Drone 11: Go from v11 = [-2.   4.5  2. ] to v22 = [-4.5  0.   2. ]
Drone 11: No complete path found
Suces_rte =  [1, 0]
Drone 11: A* done under 1.0045325756072998 sec
Drone 11: Trajectory start: 2.44 sec
Drone 11: Trajectory end: 4.44 sec
Drone 11: Trajectory duration: 2.0000000000000004 sec
---------------------------------1.46----------------------------------------
Fuck
Fuck
Fuck
Drone 12: Go from v12 = [-1.   4.5  2. ] to v22 = [-4.5  0.   2. ]
Suces_rte =  [1, 1]
Drone 12: A* done under 0.006000041961669922 sec
Drone 12: Trajectory start: 3.46 sec
Drone 12: Trajectory end: 12.16 sec
Drone 12: Trajectory duration: 8.7 sec
---------------------------------1.48----------------------------------------
Drone 13: Go from v13 = [0.  4.5 2. ] to v7 = [ 3.  -4.5  2. ]
Suces_rte =  [1, 1]
Drone 13: A* done under 0.009999990463256836 sec
Drone 13: Trajectory start: 3.48 sec
Drone 13: Trajectory end: 16.98 sec
Drone 13: Trajectory duration: 13.5 sec
---------------------------------1.5----------------------------------------
Drone 14: Go from v14 = [1.  4.5 2. ] to v32 = [4.5 1.  2. ]
Suces_rte =  [1, 1]
Drone 14: A* done under 0.006011009216308594 sec
Drone 14: Trajectory start: 3.5 sec
Drone 14: Trajectory end: 11.7 sec
Drone 14: Trajectory duration: 8.2 sec
---------------------------------4.46----------------------------------------
Drone 11: Go from v11 = [-2.   4.5  2. ] to v27 = [ 4.5 -4.   2. ]
Suces_rte =  [2, 1]
Drone 11: A* done under 0.027999401092529297 sec
Drone 11: Trajectory start: 6.46 sec
Drone 11: Trajectory end: 24.26 sec
Drone 11: Trajectory duration: 17.8 sec
---------------------------------4.84----------------------------------------
Drone 0: Go from v19 = [-4.5 -3.   2. ] to v25 = [-4.5  3.   2. ]
Suces_rte =  [2, 2]
Drone 0: A* done under 0.07734942436218262 sec
Drone 0: Trajectory start: 6.84 sec
Drone 0: Trajectory end: 19.04 sec
Drone 0: Trajectory duration: 12.2 sec
---------------------------------6.12----------------------------------------
Drone 3: Go from v2 = [-2.  -4.5  2. ] to v35 = [4.5 4.  2. ]
Suces_rte =  [2, 2]
Drone 3: A* done under 0.22845196723937988 sec
Drone 3: Trajectory start: 8.12 sec
Drone 3: Trajectory end: 26.52 sec
Drone 3: Trajectory duration: 18.4 sec
---------------------------------8.26----------------------------------------
Drone 1: Go from v20 = [-4.5 -2.   2. ] to v26 = [-4.5  4.   2. ]
Suces_rte =  [2, 2]
Drone 1: A* done under 0.016391754150390625 sec
Drone 1: Trajectory start: 10.26 sec
Drone 1: Trajectory end: 21.56 sec
Drone 1: Trajectory duration: 11.299999999999999 sec
---------------------------------8.76----------------------------------------
Drone 8: Go from v3 = [-1.  -4.5  2. ] to v2 = [-2.  -4.5  2. ]
Suces_rte =  [2, 2]
Drone 8: A* done under 0.0009999275207519531 sec
Drone 8: Trajectory start: 10.76 sec
Drone 8: Trajectory end: 12.76 sec
Drone 8: Trajectory duration: 2.0 sec
---------------------------------11.72----------------------------------------
Drone 14: Go from v32 = [4.5 1.  2. ] to v3 = [-1.  -4.5  2. ]
Suces_rte =  [2, 2]
Drone 14: A* done under 0.014999866485595703 sec
Drone 14: Trajectory start: 13.72 sec
Drone 14: Trajectory end: 27.22 sec
Drone 14: Trajectory duration: 13.499999999999998 sec
---------------------------------12.18----------------------------------------
Drone 12: Go from v22 = [-4.5  0.   2. ] to v0 = [-4.  -4.5  2. ]
Suces_rte =  [2, 2]
Drone 12: A* done under 0.007999420166015625 sec
Drone 12: Trajectory start: 14.18 sec
Drone 12: Trajectory end: 23.08 sec
Drone 12: Trajectory duration: 8.899999999999999 sec
---------------------------------12.78----------------------------------------
Drone 8: Go from v2 = [-2.  -4.5  2. ] to v22 = [-4.5  0.   2. ]
Suces_rte =  [3, 3]
Drone 8: A* done under 0.01699995994567871 sec
Drone 8: Trajectory start: 14.78 sec
Drone 8: Trajectory end: 24.68 sec
Drone 8: Trajectory duration: 9.9 sec
---------------------------------12.94----------------------------------------
Drone 4: Go from v1 = [-3.  -4.5  2. ] to v18 = [-4.5 -4.   2. ]
Drone 4: No complete path found
Suces_rte =  [2, 1]
Drone 4: A* done under 1.0034153461456299 sec
Drone 4: Trajectory start: 14.94 sec
Drone 4: Trajectory end: 16.94 sec
Drone 4: Trajectory duration: 2.0000000000000018 sec
---------------------------------13.96----------------------------------------
Drone 2: Go from v28 = [ 4.5 -3.   2. ] to v18 = [-4.5 -4.   2. ]
Suces_rte =  [2, 2]
Drone 2: A* done under 0.03899502754211426 sec
Drone 2: Trajectory start: 15.96 sec
Drone 2: Trajectory end: 32.26 sec
Drone 2: Trajectory duration: 16.299999999999997 sec
---------------------------------14.54----------------------------------------
Drone 7: Go from v16 = [3.  4.5 2. ] to v19 = [-4.5 -3.   2. ]
Suces_rte =  [2, 2]
Drone 7: A* done under 0.011999845504760742 sec
Drone 7: Trajectory start: 16.54 sec
Drone 7: Trajectory end: 31.14 sec
Drone 7: Trajectory duration: 14.600000000000001 sec
---------------------------------14.58----------------------------------------
Drone 5: Go from v34 = [4.5 3.  2. ] to v28 = [ 4.5 -3.   2. ]
Suces_rte =  [2, 2]
Drone 5: A* done under 0.008999347686767578 sec
Drone 5: Trajectory start: 16.58 sec
Drone 5: Trajectory end: 23.58 sec
Drone 5: Trajectory duration: 7.0 sec
---------------------------------16.96----------------------------------------
Drone 4: Go from v1 = [-3.  -4.5  2. ] to v6 = [ 2.  -4.5  2. ]
Suces_rte =  [3, 2]
Drone 4: A* done under 0.17133593559265137 sec
Drone 4: Trajectory start: 18.96 sec
Drone 4: Trajectory end: 29.26 sec
Drone 4: Trajectory duration: 10.3 sec
---------------------------------17.14----------------------------------------
Drone 9: Go from v17 = [4.  4.5 2. ] to v9 = [-4.   4.5  2. ]
Suces_rte =  [2, 2]
Drone 9: A* done under 0.014999866485595703 sec
Drone 9: Trajectory start: 19.14 sec
Drone 9: Trajectory end: 28.14 sec
Drone 9: Trajectory duration: 9.0 sec
---------------------------------17.16----------------------------------------
Drone 13: Go from v7 = [ 3.  -4.5  2. ] to v15 = [2.  4.5 2. ]
Suces_rte =  [2, 2]
Drone 13: A* done under 0.009003639221191406 sec
Drone 13: Trajectory start: 19.16 sec
Drone 13: Trajectory end: 32.16 sec
Drone 13: Trajectory duration: 12.999999999999996 sec
---------------------------------18.42----------------------------------------
Drone 10: Go from v8 = [ 4.  -4.5  2. ] to v7 = [ 3.  -4.5  2. ]
Suces_rte =  [2, 2]
Drone 10: A* done under 0.0013840198516845703 sec
Drone 10: Trajectory start: 20.42 sec
Drone 10: Trajectory end: 22.42 sec
Drone 10: Trajectory duration: 2.0 sec
---------------------------------19.06----------------------------------------
Drone 0: Go from v25 = [-4.5  3.   2. ] to v33 = [4.5 2.  2. ]
Suces_rte =  [3, 3]
Drone 0: A* done under 0.012000322341918945 sec
Drone 0: Trajectory start: 21.06 sec
Drone 0: Trajectory end: 34.36 sec
Drone 0: Trajectory duration: 13.3 sec
---------------------------------19.3----------------------------------------
Drone 6: Go from v23 = [-4.5  1.   2. ] to v2 = [-2.  -4.5  2. ]
Suces_rte =  [2, 2]
Drone 6: A* done under 0.0480046272277832 sec
Drone 6: Trajectory start: 21.3 sec
Drone 6: Trajectory end: 31.7 sec
Drone 6: Trajectory duration: 10.399999999999999 sec
"""


# t, cacl, tau_end

d_0 = [[ 0.02, 0.0009, 4.82],
       [ 4.84, 0.0773, 19.04],
       [19.06, 0.0120, 34.36]]

d_1 = [[ 0.04, 0.0200, 8.24],
       [ 8.26, 0.0163, 21.56]]

d_2 = [[ 0.08, 0.0179, 13.48],
       [13.96, 0.0389, 32.26]]

d_3 = [[ 0.10, 0.0019, 6.10],
       [ 6.12, 0.2284, 26.52]]

d_4 = [[ 0.12, 0.1230, 12.92],
       [12.94, 1.0034, 16.94],
       [16.96, 0.1713, 29.26]]

d_5 = [[ 0.26, 0.0110, 14.56],
       [14.58, 0.0089, 23.58]]

d_6 = [[ 0.28, 0.0240, 19.28],
       [19.30, 0.0480, 31.7]]

d_7 = [[ 0.32, 0.0094, 14.52],
       [14.54, 0.0119, 31.14]]

d_8 = [[ 0.34, 0.0069, 8.74],
       [ 8.76, 0.0009, 12.76],
       [12.78, 0.0169, 24.68]]

d_9 = [[0.36,  0.0399, 17.06],
       [17.14, 0.0149, 28.14]]

d_10 = [[ 0.40, 0.0260, 18.4],
        [18.42, 0.0013, 22.42]]

d_11 = [[ 0.44, 1.0045, 4.44],
        [ 4.46, 0.0279, 24.26]]

d_12 = [[ 1.46, 0.0060, 12.16],
        [12.18, 0.0079,  23.08]]

d_13 = [[ 1.48, 0.0099, 16.98],
        [17.16, 0.0090, 32.16]]

d_14 = [[ 1.50, 0.0060, 11.7],
        [11.72, 0.0149, 27.22]]


data = [d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11, d_12, d_13, d_14]
data = data[::-1]

sum_plan_time = 0
plan_num = 0

for d in data:
    d[-1][-1] = 20

    for plan in d:
        if plan[1]<1:
            plan_num += 1
            sum_plan_time += plan[1]

print("Avg plan time:", sum_plan_time/plan_num)


# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

wait_sum = 0
execution_sum = 0

# Plot each drone's paths
for drone_id, path_plan in enumerate(data):
    ax.barh(drone_id, path_plan[0][0], color='orange', alpha=0.8)
    wait_sum += path_plan[0][0]
    for i, plan in enumerate(path_plan):
        plan_start, calc_time, path_end = plan
        ax.barh(drone_id, path_end-(plan_start + calc_time), left=plan_start+calc_time, color='green', alpha=0.8)
        execution_sum += path_end-(plan_start + calc_time)
        ax.barh(drone_id, calc_time, left=plan_start, color='blue', alpha=0.8)
        if i>0:
            ax.barh(drone_id, plan_start-path_plan[i-1][2], left=path_plan[i-1][2], color='orange', alpha=0.8)
            wait_sum += plan_start-path_plan[i-1][2]

print("Sum wait time:", wait_sum)
print("Sum execution time:", execution_sum)
print("Wait to execution rate:", wait_sum/execution_sum)

# Add labels and formatting
ax.set_yticks(range(len(data)))
ax.set_yticklabels([f'Drone {15-i}' for i in range(len(data))])
ax.set_xlabel('Time [s]')
ax.grid(axis='x', linestyle='--', alpha=0.7)


# Legend for clarity

colors = ['blue', 'green', 'orange']
names = ['plan', 'execute', 'wait']

for i, color in enumerate(colors):
    ax.barh(0, 0, color=color, label=names[i])
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))





plt.tight_layout()
plt.show()