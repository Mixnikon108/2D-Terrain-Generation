# Author: Jorge de la Rosa @mixnikon108
# Date: 1/5/2024
# Description: This program visualizes terrain generation. 
# It initializes a single triangle and recursively subdivides it, displaying the results graphically.

import pygame
from utils import Node, Triangle, FIFO

def main():
    """Main function to setup and run the Pygame visualization loop."""
    pygame.init()  # Initialize Pygame

    # Define color constants
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # Setup the display
    screen_size = (700, 700)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Terrain Visualization")

    # Create the initial triangle
    A = Node(100, 600)
    B = Node(350, 200)
    C = Node(600, 600)
    root_triangle = Triangle(A, B, C)

    # Initialize the queue to manage triangles
    queue = FIFO()
    queue.enqueue(root_triangle)

    # Set the depth for triangle subdivisions
    depth = 6
    for _ in range((4 ** depth - 1) // 3):
        current_triangle = queue.dequeue()
        triangles = current_triangle.subdivide()
        queue.enqueue(triangles, is_list=True)

    # Main loop control
    done = False
    clock = pygame.time.Clock()

    # Main event and drawing loop
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Clear screen to white before drawing
        screen.fill(WHITE)  

        # Draw triangles
        for _ in range(queue.size()):
            current_triangle = queue.dequeue()
            current_triangle.draw(screen, BLACK)  # Draw triangle with black color
            queue.enqueue(current_triangle)  # Re-enqueue for continuous display

        pygame.display.flip()  # Update display
        clock.tick(10)  # Control frame rate

    pygame.quit()

if __name__ == "__main__":
    main()
