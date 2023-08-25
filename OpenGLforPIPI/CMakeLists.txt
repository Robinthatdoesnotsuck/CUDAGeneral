cmake_minimum_required(VERSION 3.5.0)
project(OpenGLForPooPooHeads VERSION 0.1.0 LANGUAGES CUDA C CXX)

include(CTest)
find_package(OpenGL)
find_package(GLUT)
enable_testing()
enable_language(CUDA)
add_executable(OpenGLForPooPooHeads conejito.cu)
set(obj_file_path ${CMAKE_CURRENT_SOURCE_DIR}/stanford-bunny.obj)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
configure_file(${obj_file_path} ${CMAKE_CURRENT_BINARY_DIR}/stanford-bunny.obj COPYONLY)
include(CPack)


include(CheckLanguage)
check_language(CUDA)

target_link_libraries(
    OpenGLForPooPooHeads
    glfw
    GLEW
    OpenGL
    glut
    GLU
)

