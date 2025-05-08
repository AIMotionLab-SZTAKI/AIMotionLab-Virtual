import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl

from Utils.Utils import load_scenario_and_drones, handle_dynamic_obstacles, set_rand_seed, save_drones_routes, set_anim_time, set_delay, animate_drones, check_routes_save_file, min_distance_between_drones, c_print

if __name__ == '__main__':
    # -------- LOAD DATA --------
    scenario, drones = load_scenario_and_drones(file_name="city_scenario")
    scenario.print_data()

    # -------- PREPARE FOR ANIMATION --------
    scenario.plot(plt_targets=False)
    fig = plt.gcf()

    drone_num = 4
    virtual_drone_num = 11

    landing_delay = 2

    demo_time = 30 # which after every drone will return to its home position
    animation_speed = 1
    delay_between_frames = 20 # 20 is ideal 500 is more computational friendly
    skip_animation = True
    save_animation = False
    saved_time = 20 # +dt to take into account the time to return home
    anim_save_name = f"{drone_num}_city"

    real_routes_save_name = check_routes_save_file(file_name="real_routes")
    virtual_routes_save_name = check_routes_save_file(file_name="virtual_routes")

    set_rand_seed(130010)

    # -------- ANIMATION --------
    drones = drones[:(drone_num+virtual_drone_num)]
    # Collect the routes of the drones and save the complete scenario
    real_routes = [[] for _ in range(drone_num)]
    virtual_routes = [[] for _ in range(virtual_drone_num)]

    for i, _drone in enumerate(drones):
        if i < drone_num:
            _drone.place(scenario=scenario, vertex=scenario.target_vertices[0], lock_home_vertex=True) # !!!
        else:
            _drone.color = 'orange'
            _drone.place(scenario=scenario, vertex=scenario.extra_target_vertices[-1], lock_home_vertex=True)  # !!!
            _drone.virtual = True


    def update(frame):
        if frame == 0:
            print("Plot scene")
        else:
            time_now = set_anim_time(drones, frame, delay_between_frames, animation_speed, skip_animation)

            if scenario.dynamic_obstacles is not None:
                handle_dynamic_obstacles(scenario=scenario, t=time_now, drones=drones)
                for dynamic_obs in scenario.dynamic_obstacles:
                    dynamic_obs.animate(t=time_now)
                    dynamic_obs.plot_trajectory()

            for drone in drones:
                traj_generated = False
                if demo_time > time_now >= max(drone.trajectory_final_time(), drone.wait_others):
                    print(f"---------------------------------{time_now}----------------------------------------")
                    drone.go_to_new_target(start_time=time_now, scenario=scenario, drones=drones) # !!!
                    traj_generated = True

                elif time_now >= max(drone.trajectory_final_time(), drone.wait_others) and time_now >= demo_time and not drone.returned_home:
                    print(f"---------------------------------{time_now}----------------------------------------")
                    drone.go_to_new_target(start_time=time_now, scenario=scenario, drones=drones, go_home=True)
                    traj_generated = True

                elif time_now >= (drone.trajectory_final_time()+landing_delay) and time_now >= demo_time and not drone.landed and drone.returned_home:
                    # Land drone
                    print(f"---------------------------------{time_now}----------------------------------------")
                    drone.land(scenario=scenario, t=time_now)

                if traj_generated:
                    set_delay(drones, drone.ID, time_now, zero_delay=skip_animation)

                    # Store the roure for save
                    drone.route[:, 0] += time_now
                    if drone.virtual:
                        virtual_routes[drone.ID-drone_num].append({"Start": drone.start_vertex,
                                                                   "Route": drone.route})
                    else:
                        real_routes[drone.ID].append({"Start": drone.start_vertex,
                                                      "Route": drone.route})

                    drone.plot_trajectory()
                    continue

            animate_drones(drones, time_now)
            min_dist = min_distance_between_drones(drones, time_now)
            if min_dist < 2*drone.radius:
                c_print(f"\n------------------------------------\n"
                        f"Collision warning: Minimum distance between drones is {min_dist} m\n"
                        f"------------------------------------\n", "red")
            save_drones_routes(drones, real_routes, real_routes_save_name)
            save_drones_routes(drones, virtual_routes, virtual_routes_save_name)


    mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\Mate\Desktop\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
    anim = animation.FuncAnimation(fig, update, interval=delay_between_frames, blit=False, cache_frame_data=False,
                                   save_count=int(saved_time / (delay_between_frames / 1000)))
    if save_animation:
        f = r"Saves/Animations/"+anim_save_name+".mp4"
        writermp4 = animation.FFMpegWriter(fps=int(1/(delay_between_frames/1000)))
        anim.save(f, writer=writermp4)
    else:
        plt.show()
