import matplotlib.pyplot as plt
import numpy as np
import random

# Define the function g(x) (can plug in whatever values in here to graph whatever function you would like)
def g(x):
  return (0.0051 * x**5) - (0.1367 * x**4) + (1.24 * x**3) - (4.456 * x**2) + (5.66 * x) - 0.287

# Get step size from the user
step_size = float(input("Enter the step size: "))

# Create the x values with a step size from user
x_values = np.arange(0, 10 + step_size, step_size)

# Calculate the y values using the g(x) function
y_values = g(x_values)

# Create the plot
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Graph of g(x)')
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
        if g(neighbor) <= g(current):
            return g(current)
        else:
            current = neighbor

# Now lets implement the random start function to plug into the hill_climb as well as the algorithim to run it
# function random start() (returns max value from the random starts)
    # int max
    # for i = 1 to 20 do
        # random = random number within x range for graph
        # temp = hill_climb(random)
        # if max < temp:
            # max = temp

def random_start():
    max = 0
    for i in range(20):
        random_x = random.randint(0, 10)
        temp = hill_climb(random_x)
        if max < temp:
            max = temp
    return max

# Call random_start
max = random_start()

# Add text boxes to display results of random/conventional hill climb
plt.text(0, -4.5, "Random Restart maximum (20 random starts) at y = " + str(round(max, 2)))
plt.text(0, -5, "Local maximum (starting at x = 0) at y = " + str(round(hill_climb(0), 2)))

# Show the plot
plt.show()
