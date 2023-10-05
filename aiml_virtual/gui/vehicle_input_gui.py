from curses import window
import tkinter as tk

# Dropdown menu options
VEHICLE_TYPE_OPTIONS = [
    "Virtual crazyflie",
    "Virtual bumblebee",
    "Virtual bb with hook",
    "Mocap crazyflie",
    "Mocap bumblebee",
    "Mocap bb with hook",
    "Virtual Fleet1Tenth",
    "Virtual F1Tenth with rod",
    "Mocap Fleet1Tenth",
    "Mocap F1Tenth with rod"
]


class VehicleInputGui:
    
    def __init__(self):
        self.window = tk.Tk()

        self.window.title("Add Vehicle")

        self.needs_new_vehicle = False

        self.vehicle_type = ""
        self.position = ""
        self.quaternion = ""
        
        tk.Label(self.window, text="Vehicle Type ").grid(row=0)
        tk.Label(self.window, text="Position ").grid(row=1)
        tk.Label(self.window, text="Quaternion ").grid(row=2)

        self.vehicle_type_selected = tk.StringVar()
        self.vehicle_type_selected.set(VEHICLE_TYPE_OPTIONS[0])

        self.opt_vehicle_type = tk.OptionMenu(self.window, self.vehicle_type_selected, *VEHICLE_TYPE_OPTIONS)
        self.entry_position = tk.Entry(self.window)
        self.entry_quaternion = tk.Entry(self.window)
        self.entry_quaternion.insert(0, "1 0 0 0")

        self.opt_vehicle_type.grid(row=0, column=1)
        self.entry_position.grid(row=1, column=1)
        self.entry_quaternion.grid(row=2, column=1)

        tk.Button(self.window, text ="Ok", command = self.btnOk_on_press).grid(row=3, column=1)
        self.window.bind('<Return>', lambda event: self.btnOk_on_press())

        
    def btnOk_on_press(self):
        self.vehicle_type, self.position, self.quaternion =\
            self.vehicle_type_selected.get(), self.entry_position.get(), self.entry_quaternion.get()
        self.needs_new_vehicle = True
        self.window.quit()
        self.window.destroy()

    def on_closing(self):
        self.window.quit()
        self.window.destroy()
    
    def show(self):
        #self.window.focus_force()
        self.opt_vehicle_type.focus_force()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
