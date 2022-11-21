from curses import window
import tkinter as tk

class DroneNameGui:

    def __init__(self, drone_num=4, drone_names=[]):
        if drone_num <= 0:
            return
        
        self.window = tk.Tk()

        self.window.title("Name drones")

        self.entries = []
        self.drone_names = []

        for i in range(drone_num):
            tk.Label(self.window, text="Drone " + str(i + 1) + " name: ").grid(row=i)
            self.entries.append(tk.Entry(self.window))
            self.entries[i].grid(row=i, column=1)
            if len(drone_names) == drone_num:
                self.entries[i].insert(0, drone_names[i])
            else:
                self.entries[i].insert(0, "cf" + str(i + 1))


        tk.Button(self.window, text ="Ok", command = self.btnOk_on_press).grid(row=drone_num, column=1)
        self.window.bind('<Return>', lambda event: self.btnOk_on_press())

        
    def btnOk_on_press(self):
        for i in range(len(self.entries)):
            self.drone_names.append(self.entries[i].get())
        self.window.quit()
        self.window.destroy()
        #print(self.drone_names)

    def on_closing(self):
        for i in range(len(self.entries)):
            self.drone_names.append(self.entries[i].get())
        self.window.quit()
        self.window.destroy()


    def show(self):
        #self.window.focus_force()
        self.entries[0].focus_force()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

