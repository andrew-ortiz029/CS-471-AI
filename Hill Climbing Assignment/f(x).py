import matplotlib.pyplot as plt
import numpy as np

# Define the function f(x) (can plug in whatever values in here to graph whatever function you would like)
def f(x):
  return (2 - x**2)

# Get step size from the user
step_size = float(input("Enter the step size: "))

# Create the x values with a step size from user
x_values = np.arange(-5, 5 + step_size, step_size)

# Calculate the y values using the f(x) function
y_values = f(x_values)

# Create the plot
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x) = 2 - x^2 Step-Size= ' + str(step_size))
plt.grid(True)

# Implement hill climb algorithim
# function Hill-Climbing(graph) (returns a local maxiumum)
    # current = graph Starting Point
    # while true do
        # neighbor = next step in direction
        # if value(neighbor) <= Value(current)
            # return current y
        # else
            # current = neighbor

def hill_climb(start):
    current = start
    while True:
        neighbor = current + step_size
        if f(neighbor) <= f(current):
            return f(current)
        else:
            current = neighbor

# Add text box to display local maximum
plt.text(-1.85, -15, "Local maximum at y = " + str(round(hill_climb(0), 2)))

# Show the plot
plt.show()
