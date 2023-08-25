cmake_minimum_required(VERSION 3.0.0)
project(CudaHello VERSION 0.1.0 LANGUAGES CUDA C CXX)

include(CTest)
enable_testing()
# Specify the CSV file path
set(csv_file_path ${CMAKE_CURRENT_SOURCE_DIR}/LifeExpectancyData.csv)
set(csv_file_path ${CMAKE_CURRENT_SOURCE_DIR}/Iris.csv)
# Copy the CSV file to the build directory
configure_file(${csv_file_path} ${CMAKE_CURRENT_BINARY_DIR}/LifeExpectancyData.csv COPYONLY)
configure_file(${csv_file_path} ${CMAKE_CURRENT_BINARY_DIR}/Iris.csv COPYONLY)

enable_language(CUDA)
add_executable(CudaHello lifeExp.cu)

include(CheckLanguage)
check_language(CUDA)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})



include(CPack)


if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

