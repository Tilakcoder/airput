import tkinter as tk

class CircleWindow:
    def __init__(self):
        # Initialize the tkinter window
        self.root = tk.Tk()
        self.root.title("Circle on Screen")

        # Make the window full-screen
        self.root.attributes("-fullscreen", True)

        # Create a canvas to draw on
        self.canvas = tk.Canvas(self.root, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.canvas.pack()

        # Flag to indicate if the window should be closed
        self.should_close = False

    def draw_circle(self, x, y, radius):
        # Function to draw a circle on the canvas
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="red", width=2)

    def start(self):
        # Start the GUI main loop
        self.root.after(100, self.check_close)  # Check for close request every 100 milliseconds
        self.root.mainloop()

    def check_close(self):
        # Check if the window should be closed
        if self.should_close:
            self.root.attributes("-fullscreen", False)
            self.root.destroy()  # Destroy the window
        else:
            self.root.after(100, self.check_close)  # Continue checking for close request

    def close(self):
        # Set the flag to indicate that the window should be closed
        self.should_close = True
