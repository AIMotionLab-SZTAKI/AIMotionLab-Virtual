from curses import window
import tkinter as tk

class VehicleNameGui:

    def __init__(self, vehicle_labels=[], vehicle_names=[]):
        if len(vehicle_labels) != len(vehicle_names):
            return
        
        self.window = tk.Tk()

        self.window.title("Name vehicles")

        self.entries = []
        self.vehicle_names = []
        self.initial_names = vehicle_names

        if len(vehicle_names) != len(vehicle_labels):
            print("[VehicleNameGui.__init__()] Different number of vehicle names and number of vehicle labels")
            return

        for i in range(len(vehicle_labels)):
            tk.Label(self.window, text=vehicle_labels[i] + ": ").grid(row=i)
            self.entries.append(tk.Entry(self.window))
            self.entries[i].grid(row=i, column=1)
            if len(vehicle_names) == len(vehicle_labels):
                self.entries[i].insert(0, vehicle_names[i])
            else:
                self.entries[i].insert(0, "cf" + str(i + 1))


        tk.Button(self.window, text ="Ok", command = self.btnOk_on_press).grid(row=len(vehicle_labels), column=1)
        self.window.bind('<Return>', lambda event: self.btnOk_on_press())

        
    def btnOk_on_press(self):
        for i in range(len(self.entries)):
            self.vehicle_names.append(self.entries[i].get())
        self.window.quit()
        self.window.destroy()
        #print(self.vehicle_names)

    def on_closing(self):
        #for i in range(len(self.entries)):
        #    self.vehicle_names.append(self.entries[i].get())
        self.vehicle_names = self.initial_names
        self.window.quit()
        self.window.destroy()


    def show(self):
        #self.window.focus_force()
        self.entries[0].focus_force()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

