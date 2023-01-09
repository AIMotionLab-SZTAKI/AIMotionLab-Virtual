from curses import window
import tkinter as tk


COLORS = {
    "Blue" : "0.1 0.1 0.9 1.0",
    "Green" : "0.1 0.9 0.1 1.0",
    "Red" : "0.9 0.1 0.1 1.0",
    "Yellow" : "0.7 0.7 0.1 1.0"
}


class CargoInputGui:
    
    def __init__(self):
        self.window = tk.Tk()

        self.window.title("Add load")

        self.color = ""
        self.size = ""
        self.mass = ""
        self.position = ""
        self.quaternion = ""
        
        tk.Label(self.window, text="Color ").grid(row=0)
        tk.Label(self.window, text="Size ").grid(row=1)
        tk.Label(self.window, text="Mass ").grid(row=2)
        tk.Label(self.window, text="Position ").grid(row=3)
        tk.Label(self.window, text="Quaternion ").grid(row=4)

        self.color_selected = tk.StringVar()
        self.color_selected.set(list(COLORS.keys())[0])

        self.opt_color = tk.OptionMenu(self.window, self.color_selected, *COLORS.keys())
        self.entry_size = tk.Entry(self.window)
        self.entry_mass = tk.Entry(self.window)
        self.entry_position = tk.Entry(self.window)
        self.entry_quaternion = tk.Entry(self.window)
        self.entry_quaternion.insert(0, "1 0 0 0")

        self.opt_color.grid(row=0, column=1)
        self.entry_size.grid(row=1, column=1)
        self.entry_mass.grid(row=2, column=1)
        self.entry_position.grid(row=3, column=1)
        self.entry_quaternion.grid(row=4, column=1)

        tk.Button(self.window, text ="Ok", command = self.btnOk_on_press).grid(row=5, column=1)
        self.window.bind('<Return>', lambda event: self.btnOk_on_press())

        
    def btnOk_on_press(self):
        self.color, self.mass, self.size, self.position, self.quaternion =\
            COLORS[self.color_selected.get()], self.entry_mass.get(), self.entry_size.get(),\
                self.entry_position.get(), self.entry_quaternion.get()
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
