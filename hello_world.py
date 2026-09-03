import matplotlib.pyplot as plt

def create_hello_world_plot(filename="hello_world.png"):
    # Create a simple figure
    plt.figure()

    # Add text in the center
    plt.text(0.5, 0.5, "Hello, World!", fontsize=20, ha='center')

    # Remove axes for a clean look
    plt.axis('off')

    # Save the figure in current directory
    plt.savefig(filename)

    # Close the plot to free memory
    plt.close()

# Call the function
create_hello_world_plot()