import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import math

# Create graphs and axis
fig, ax = plt.subplots()

# Test cases list of lists containing tuples
test_cases = [[(1, 3, 0.7), (2, 3, 0.4), (3, 3, 0.9)], 
              [(1.5, 1.5, 1.3), (4, 4, 0.7)], 
              [(0.5, 0.5, 0.5), (1.5, 1.5, 1.1), (0.7, 0.7, 0.4), (4, 4, 0.7)],
              [(2.5, 2.5, 1.5), (2.5, 2.5, 1), (2.5, 2.5, .5)]]
index = 0

# function that takes in a list and adds the test cases to the graph
def graph_circles(circles):
    ax.clear()
    for x, y, radius in circles:
        
        # Create a circle for each circle in the test cases
        circle = patches.Circle((x, y), radius, edgecolor='black', facecolor='none')
        
        # Add the circle to the axis
        ax.add_patch(circle)
        
        # Set limits
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)

        # Set equal scaling
        ax.set_aspect('equal')

        # Add grid for better visualization
        ax.grid(True)

        # Send to cluster_check for T/F output
        cluster_check(circles)
        plt.title("Clustered: True")

        # Draw out the graph
        plt.draw()

# Check Graph for Cluster
def cluster_check(circles):
    # To check if two circles are intersected, check the distance from their centers and compare to the sum of their radius
        # A few conditions will arise but we'll only check for them being intersected
        # 1. If distance < radius1 + radius 2 (they intersect or touch)
        # 2. Anything else we don't care about and is not a cluster
    
    # I will be using a modified DFT algorithm to keep the number of comparisons to a minimum and to ensure the graph is connected
        # Modified to check if every circle has been visited and return true instead of continuing to traverse the whole graph and its connections
        # Modified to return false instantly on the first iteration of there's no connections

    # Vars for comparison
    x1, x2, y1, y2, r1, r2

    # List to keep track of visited circles 
    visited = [False] * len(circles)

    # Stack to keep track of DFT
    stack = []

    # iterate through circle 0 connections and initialize the stack with that 
    i = 0
    while i < len(circles):
        

        i += 1

    # Main DFT control will run as long as there's a connection in the stack or until returned
    while stack: # run until returned
        i = 0
        

    # If all circles have been visited then return true as the graph is clustered
    if visited == [True] * len(circles):
        return True
    
    # Equation for computing distance 
    distance = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    


# Event handler for the 'Next' button
def next_graph(event):
    global index
    graph_circles(test_cases[index])
    if index == 3:
        index = 0
    else:
        index = index + 1 # next test case

# Create a button for next graph
next_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
next_button = Button(next_button_ax, 'Next')
next_button.on_clicked(next_graph)

# Show the plot
plt.show()