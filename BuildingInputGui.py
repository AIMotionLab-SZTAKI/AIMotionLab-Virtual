from curses import window
import tkinter as tk

def btnOk_on_press(gui):
    gui.building, gui.position, gui.quaternion =\
        gui.entry_building.get(), gui.entry_position.get(), gui.entry_quaternion.get()
    gui.window.destroy()

class BuildingDataGui:
    
    def __init__(self):
        self.window = tk.Tk()

        self.building = ""
        self.position = ""
        self.quaternion = ""
        
        tk.Label(self.window, text="Building").grid(row=0)
        tk.Label(self.window, text="Position").grid(row=1)
        tk.Label(self.window, text="Quaternion").grid(row=2)

        self.entry_building = tk.Entry(self.window)
        self.entry_position = tk.Entry(self.window)
        self.entry_quaternion = tk.Entry(self.window)
        self.entry_quaternion.insert(0, "1 0 0 0")

        self.entry_building.grid(row=0, column=1)
        self.entry_position.grid(row=1, column=1)
        self.entry_quaternion.grid(row=2, column=1)

        tk.Button(self.window, text ="Ok", command = lambda: btnOk_on_press(self)).grid(row=3, column=1)
        self.window.bind('<Return>', lambda event: btnOk_on_press(self))
    
    def show(self):
        #self.window.focus_force()
        self.entry_building.focus_force()
        self.window.mainloop()
