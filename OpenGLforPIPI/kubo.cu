#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

using namespace std;

// PI definition
#define M_PI 3.14159265358979323846
// the main window size
GLint wWindow = 500;
GLint hWindow = 500;

float angle = 0.0;

GLfloat vertices[] = {
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f
};

GLubyte indices[] = {
    0, 1, 2, 3,
    3, 2, 6, 7,
    7, 6, 5, 4,
    4, 5, 1, 0,
    0, 3, 7, 4,
    1, 5, 6, 2
};

GLuint vbo;
struct cudaGraphicsResource* vbo_resource;
float* d_vertices = nullptr;

__global__ void calculateVertices(float* vertices, int numVertices, float angle) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numVertices) {
        // Perform some calculation on the vertices
        // For example, rotate around the y-axis
        float x = vertices[idx * 3];
        float z = vertices[idx * 3 + 2];
        vertices[idx * 3] = x * cos(angle) - z * sin(angle);
        vertices[idx * 3 + 2] = x * sin(angle) + z * cos(angle);
    }
}

void cleanup() {
    if (d_vertices != nullptr) {
        cudaGraphicsUnmapResources(1, &vbo_resource, 0);
        cudaGraphicsUnregisterResource(vbo_resource);
        d_vertices = nullptr;
    }
}

void initArray() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 5.0,
        0.0, 0.0, -1.0,
        0.0f, 1.0f, 0.0f);
    glRotatef(angle, 0.0f, 1.0f, 0.0f);

    // Enable lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Define a light
    GLfloat light_position[] = { 0.0, 0.0, 1.0, 0.0 };
    GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);

    // Set the material properties for the cubes
    GLfloat mat_diffuse[] = { 1.0f, 0.5f, 0.0f, 1.0f };
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);

    glEnableClientState(GL_VERTEX_ARRAY);
}

void renderSceneGPU(void) {
    // Map OpenGL buffer object for writing from CUDA
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, vbo_resource);

    // Execute the kernel
    int numVertices = sizeof(vertices) / (3 * sizeof(float));
    dim3 dimBlock(256);
    dim3 dimGrid((numVertices + dimBlock.x - 1) / dimBlock.x);
    calculateVertices << <dimGrid, dimBlock >> > (d_vertices, numVertices, angle);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);

    // Call initialization parameters
    initArray();

    // Render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, indices);
    glDisableClientState(GL_VERTEX_ARRAY);

    angle += 0.5f;
    glutSwapBuffers();
}

void changeSize(int w, int h) {
    if (h == 0)
        h = 1;
    float ratio = 1.0 * w / h;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    gluPerspective(45, ratio, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
}

void renderSceneCPU(void) {
    // Call initialization parameters
    initArray();

    // Draw a cube
    glVertexPointer(3, GL_FLOAT, 0, vertices);
    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, indices);
    glDisableClientState(GL_VERTEX_ARRAY);

    angle += 0.5f;
    glutSwapBuffers();
}

int window1, window2;

void idleFunc(void) {
    glutSetWindow(window1);
    glutPostRedisplay();
    glutSetWindow(window2);
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

    // Create window for CPU
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(wWindow, hWindow);
    window1 = glutCreateWindow("3D - Rotating Cube (CPU)");
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(renderSceneCPU);
    glutReshapeFunc(changeSize);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Create window for GPU
    glutInitWindowPosition(600, 100);
    glutInitWindowSize(wWindow, hWindow);
    window2 = glutCreateWindow("3D - Rotating Cube (GPU)");
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(renderSceneGPU);
    glutReshapeFunc(changeSize);

    // Create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Initialize CUDA context
    cudaFree(0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    glutIdleFunc(idleFunc);

    glutMainLoop();

    // Cleanup
    cleanup();

    return 0;
}
