from curses import window
from enum import Enum
import tkinter as tk
from classes.payload import PAYLOAD_TYPES


COLORS = {
    "Blue" : "0.1 0.1 0.9 1.0",
    "Green" : "0.1 0.9 0.1 1.0",
    "Red" : "0.9 0.1 0.1 1.0",
    "Yellow" : "0.7 0.7 0.1 1.0",
    "Black" : "0.1 0.1 0.1 1.0"
}


class PayloadInputGui:
    
    def __init__(self):
        self.window = tk.Tk()

        self.window.title("Add load")

        self.needs_new_payload = False

        self.is_mocap_int_var = tk.IntVar()
        self.is_mocap = False
        self.color = ""
        self.size = ""
        self.mass = ""
        self.position = ""
        self.quaternion = ""
        self.type = None
        
        tk.Label(self.window, text="Is mocap ").grid(row=0)
        tk.Label(self.window, text="Type ").grid(row=1)
        tk.Label(self.window, text="Color ").grid(row=2)
        tk.Label(self.window, text="Size ").grid(row=3)
        tk.Label(self.window, text="Mass ").grid(row=4)
        tk.Label(self.window, text="Position ").grid(row=5)
        tk.Label(self.window, text="Quaternion ").grid(row=6)

        self.color_selected = tk.StringVar()
        self.color_selected.set(list(COLORS.keys())[0])

        self.type_selected = tk.StringVar()
        self.type_selected.set(PAYLOAD_TYPES.Box.value)

        self.check_button_is_mocap = tk.Checkbutton(self.window, variable=self.is_mocap_int_var, command=self.check_button_is_mocap_on_change)
        self.opt_types = tk.OptionMenu(self.window, self.type_selected, *[e.value for e in PAYLOAD_TYPES], command=self.opt_types_on_change)
        self.opt_color = tk.OptionMenu(self.window, self.color_selected, *COLORS.keys())
        self.entry_size = tk.Entry(self.window)
        self.entry_mass = tk.Entry(self.window)
        self.entry_position = tk.Entry(self.window)
        self.entry_quaternion = tk.Entry(self.window)

        if self.is_mocap:
            self.check_button_is_mocap.select()
        else:
            self.check_button_is_mocap.deselect()

        self.entry_quaternion.insert(0, "1 0 0 0")

        self.check_button_is_mocap.grid(row=0, column=1)
        self.opt_types.grid(row=1, column=1)
        self.opt_color.grid(row=2, column=1)
        self.entry_size.grid(row=3, column=1)
        self.entry_mass.grid(row=4, column=1)
        self.entry_position.grid(row=5, column=1)
        self.entry_quaternion.grid(row=6, column=1)

        self.check_button_is_mocap_on_change()

        tk.Button(self.window, text ="Ok", command = self.btnOk_on_press).grid(row=7, column=1)
        self.window.bind('<Return>', lambda event: self.btnOk_on_press())

    
    def check_button_is_mocap_on_change(self):
        if self.is_mocap_int_var.get() == 1:
            self.entry_mass.config(state= "disabled")
            #print("disabled entry")
        else:
            self.entry_mass.config(state= "normal")
    
    def opt_types_on_change(self, choice):
        if self.type_selected.get() == PAYLOAD_TYPES.Teardrop.value:
            self.entry_size.config(state= "disabled")
        elif self.type_selected.get() == PAYLOAD_TYPES.Box.value:
            self.entry_size.config(state="normal")


        
    def btnOk_on_press(self):
        self.is_mocap, self.color, self.mass, self.size, self.position, self.quaternion =\
            self.is_mocap_int_var.get() == 1, \
            COLORS[self.color_selected.get()], self.entry_mass.get(), self.entry_size.get(),\
            self.entry_position.get(), self.entry_quaternion.get()
        type_val = self.type_selected.get()
        for t in PAYLOAD_TYPES:
            if t.value == type_val:
                self.type = t
        self.needs_new_payload = True
        self.window.quit()
        self.window.destroy()

    def on_closing(self):
        self.window.quit()
        self.window.destroy()
    
    def show(self):
        #self.window.focus_force()
        self.entry_size.focus_force()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
