from curses import window
import tkinter as tk

# Dropdown menu options
DRONE_TYPE_OPTIONS = [
    "Virtual",
    "Virtual with hook",
    "Real",
    "Real with hook"
]


class DroneInputGui:
    
    def __init__(self):
        self.window = tk.Tk()

        self.window.title("Add drone")

        self.drone_type = ""
        self.position = ""
        self.quaternion = ""
        
        tk.Label(self.window, text="Drone Type ").grid(row=0)
        tk.Label(self.window, text="Position ").grid(row=1)
        tk.Label(self.window, text="Quaternion ").grid(row=2)

        self.drone_type_selected = tk.StringVar()
        self.drone_type_selected.set(DRONE_TYPE_OPTIONS[0])

        self.opt_drone_type = tk.OptionMenu(self.window, self.drone_type_selected, *DRONE_TYPE_OPTIONS)
        self.entry_position = tk.Entry(self.window)
        self.entry_quaternion = tk.Entry(self.window)
        self.entry_quaternion.insert(0, "1 0 0 0")

        self.opt_drone_type.grid(row=0, column=1)
        self.entry_position.grid(row=1, column=1)
        self.entry_quaternion.grid(row=2, column=1)

        tk.Button(self.window, text ="Ok", command = self.btnOk_on_press).grid(row=3, column=1)
        self.window.bind('<Return>', lambda event: self.btnOk_on_press())

        
    def btnOk_on_press(self):
        self.drone_type, self.position, self.quaternion =\
            self.drone_type_selected.get(), self.entry_position.get(), self.entry_quaternion.get()
        self.window.quit()
        self.window.destroy()

    def on_closing(self):
        self.window.quit()
        self.window.destroy()
    
    def show(self):
        #self.window.focus_force()
        self.opt_drone_type.focus_force()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
