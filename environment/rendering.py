from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class Renderer:
    def __init__(self):
        self.window_width = 800
        self.window_height = 600
        self.exercise_progress = 0
        self.posture_correctness = True

    def init_gl(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def draw_human_model(self):
        glColor3f(0.0, 0.0, 1.0)  # Blue color for the human model
        glBegin(GL_LINES)
        # Draw a simple stick figure for demonstration
        glVertex2f(0, 0)  # Head
        glVertex2f(0, -0.5)  # Body
        glVertex2f(0, -0.5)  # Left Arm
        glVertex2f(-0.5, -0.5)  # Right Arm
        glVertex2f(0, -0.5)  # Left Leg
        glVertex2f(-0.5, -1.0)  # Right Leg
        glEnd()

        # Draw posture correctness indicator
        if self.posture_correctness:
            glColor3f(0.0, 1.0, 0.0)  # Green for correct posture
        else:
            glColor3f(1.0, 0.0, 0.0)  # Red for incorrect posture
        glBegin(GL_QUADS)
        glVertex2f(-0.8, 0.8)
        glVertex2f(-0.6, 0.8)
        glVertex2f(-0.6, 0.6)
        glVertex2f(-0.8, 0.6)
        glEnd()

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self.draw_human_model()
        glutSwapBuffers()

    def update_progress(self, progress, correctness):
        self.exercise_progress = progress
        self.posture_correctness = correctness

    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.window_width, self.window_height)
        glutCreateWindow("Rehabilitation Assistant Visualization")
        self.init_gl()
        glutDisplayFunc(self.display)
        glutIdleFunc(self.display)
        glutMainLoop()