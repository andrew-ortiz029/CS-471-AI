import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

# Create graphs and axis
fig, ax = plt.subplots()

# Test cases list of lists containing tuples
test_cases = [[(1, 3, 0.7), (2, 3, 0.4), (3, 3, 0.9)]]
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

        # Add grid for better visualization (optional)
        ax.grid(True)
        plt.draw()

def next_graph(event):
    global index
    graph_circles(test_cases[index])
    index = index + 1 # next test case

# Create a button for next graph
next_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
next_button = Button(next_button_ax, 'Next')
next_button.on_clicked(next_graph)

# Show the plot
plt.show()
