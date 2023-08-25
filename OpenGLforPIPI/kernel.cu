#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

using namespace std;

// PI definition
#define M_PI 3.14159265358979323846
// the main window size
GLint wWindow = 500;
GLint hWindow = 500;

// Vertex data for the triangle
GLfloat vertices[] = {
    0.0f,  0.5f, 0.0f, // Top vertex
   -0.5f, -0.5f, 0.0f, // Bottom left vertex
    0.5f, -0.5f, 0.0f  // Bottom right vertex
};

void renderScene(void) {
    // Set the background color
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set the triangle color to orange
    glColor3f(1.0f, 0.5f, 0.0f);

    // Draw the triangle
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vertices);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
}

int main(int argc, char** argv) {
    // Initialize GLUT and create window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(wWindow, hWindow);
    glutCreateWindow("Hello Triangle");

    GLenum err = glewInit(); // check errors
    if (GLEW_OK != err) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }

    // Register callbacks
    glutDisplayFunc(renderScene);

    // Enter GLUT event processing cycle
    glutMainLoop();

    return 0;
}
