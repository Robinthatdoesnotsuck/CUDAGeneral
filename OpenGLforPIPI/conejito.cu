#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

std::vector<float> vertices;
std::vector<float> normals;
std::vector<unsigned int> indices;

// the main window size
GLint wWindow = 800;
GLint hWindow = 800;
float angle = 0.0f;
float zoom = -0.3f;
float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f;

GLuint vbo;
struct cudaGraphicsRsource* vbo_rsource;

float* d_vertices = nullptr;
float* d_normals = nullptr;

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

__global__ void calculateNormals(float* normals, int numNormals, float angle) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numNormals) {
        // Perform some calculation on the vertices
        // For example, rotate around the y-axis
        float x = normals[idx * 3];
        float z = normals[idx * 3 + 2];
        normals[idx * 3] = x * cos(angle) - z * sin(angle);
        normals[idx * 3 + 2] = x * sin(angle) + z * cos(angle);
    }
}
void loadOBJ(const char* path) {
    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            centerX += x;
            centerY += y;
            centerZ += z;
        }
        else if (prefix == "f") {
            unsigned int a, b, c;
            iss >> a >> b >> c;
            indices.push_back(a - 1);
            indices.push_back(b - 1);
            indices.push_back(c - 1);
        }
    }

    // Calculate the center of the model
    centerX /= (vertices.size() / 3);
    centerY /= (vertices.size() / 3);
    centerZ /= (vertices.size() / 3);

    // Translate the model so that its center is at the origin
    for (size_t i = 0; i < vertices.size(); i += 3) {
        vertices[i] -= centerX;
        vertices[i + 1] -= centerY;
        vertices[i + 2] -= centerZ;
    }

    // Calculate normals
    normals.resize(vertices.size(), 0.0f);
    for (size_t i = 0; i < indices.size(); i += 3) {
        unsigned int a = indices[i];
        unsigned int b = indices[i + 1];
        unsigned int c = indices[i + 2];

        float ax = vertices[a * 3] - vertices[b * 3];
        float ay = vertices[a * 3 + 1] - vertices[b * 3 + 1];
        float az = vertices[a * 3 + 2] - vertices[b * 3 + 2];

        float bx = vertices[b * 3] - vertices[c * 3];
        float by = vertices[b * 3 + 1] - vertices[c * 3 + 1];
        float bz = vertices[b * 3 + 2] - vertices[c * 3 + 2];

        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;

        normals[a * 3] += nx;
        normals[a * 3 + 1] += ny;
        normals[a * 3 + 2] += nz;

        normals[b * 3] += nx;
        normals[b * 3 + 1] += ny;
        normals[b * 3 + 2] += nz;

        normals[c * 3] += nx;
        normals[c * 3 + 1] += ny;
        normals[c * 3 + 2] += nz;
    }

    for (size_t i = 0; i < normals.size(); i += 3) {
        float length = std::sqrt(normals[i] * normals[i] + normals[i + 1] * normals[i + 1] + normals[i + 2] * normals[i + 2]);
        normals[i] /= length;
        normals[i + 1] /= length;
        normals[i + 2] /= length;
    }
}

void display() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    GLfloat light_position[] = { 1.0f, 1.0f, 1.0f, 0.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glTranslatef(0.0f, 0.0f, zoom);
    glRotatef(angle, 0.0f, 1.0f, 0.0f);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
    glNormalPointer(GL_FLOAT, 0, &normals[0]);

    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glutSwapBuffers();
}

void idle() {
    angle += 0.1f;
    if (angle > 360.0f) {
        angle -= 360.0f;
    }
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    if (key == '+') {
        zoom += 0.1f;
    }
    else if (key == '-') {
        zoom -= 0.1f;
    }
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)width / (float)height, 0.1, 100.0);
}


void renderSceneGPU(void) {
    // Calculate Vertices
    // calculate normals
}

void cleanup() {

}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(wWindow, hWindow);
    glutCreateWindow("OpenGL - Standford Bunny");

    glewInit();

    loadOBJ("stanford-bunny.obj");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    glutMainLoop();

    return 0;
}
