#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <chrono>


using namespace std;

struct Iris {
    int id;
    float sepalLength;
    float sepalWidth;
    float petalLength;
    float petalWidth;
    int species;
};

void loadDataset(vector<Iris>& d) {
    ifstream file("Iris.csv");
    string line;
    map<string, int> speciesMap;
    int speciesCounter = 0;

    // Skip header line
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;

        Iris iris;

        getline(ss, cell, ',');
        iris.id = stoi(cell);

        getline(ss, cell, ',');
        iris.sepalLength = stof(cell);

        getline(ss, cell, ',');
        iris.sepalWidth = stof(cell);

        getline(ss, cell, ',');
        iris.petalLength = stof(cell);

        getline(ss, cell, ',');
        iris.petalWidth = stof(cell);

        getline(ss, cell, ',');
        if (speciesMap.find(cell) == speciesMap.end()) {
            speciesMap[cell] = speciesCounter++;
        }
        iris.species = speciesMap[cell];

        d.push_back(iris);
    }
}

void printPreprocessedDataset(vector<Iris>& d) {
    for (const Iris& iris : d) {
        if (iris.id < 10) {
            cout << "ID: 0" << iris.id;
        }
        else {
            cout << "ID: " << iris.id;
        }

        cout << ", \tSepal Length: " << iris.sepalLength
            << ", \tSepal Width: " << iris.sepalWidth
            << ", \tPetal Length: " << iris.petalLength
            << ", \tPetal Width: " << iris.petalWidth
            << ", \tSpecies: " << iris.species << endl;
    }
}

float mean_absolute_error(float* y_true, float* y_pred, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += abs(y_true[i] - y_pred[i]);
    }
    return sum / n;
}

float mean_squared_error(float* y_true, float* y_pred, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / n;
}

float root_mean_squared_error(float* y_true, float* y_pred, int n) {
    return sqrt(mean_squared_error(y_true, y_pred, n));
}

// CUDA Kernel function to perform multivariate linear regression
__global__ void linear_regression(float* x1, float* x2, float* x3, float* y, float* m1, float* m2, float* m3, float* c) {
    int idx = threadIdx.x;
    float x1_val = x1[idx];
    float x2_val = x2[idx];
    float x3_val = x3[idx];
    float y_val = y[idx];

    float prediction = (*m1) * x1_val + (*m2) * x2_val + (*m3) * x3_val + (*c);
    float error = y_val - prediction;

    // atomicAdd is a function provided by CUDA that performs an atomic (i.e., thread-safe) addition operation.
    // In a parallel computing environment like a GPU, multiple threads might try to update the same memory location simultaneously.
    // This can lead to a race condition, where the final result depends on the order in which the threads execute.
    //
    // Atomic operations prevent these race conditions by ensuring that the read - modify - write
    // operations they perform are done as a single, uninterruptible unit.
    //
    // The learning rate is a hyperparameter that determines the step size at each iteration
    // while moving toward a minimum of a loss function.
    // It controls how much we are adjusting the weights of our network with respect to the loss gradient.
    // Sometimes you have to reduce learning rate
    float learning_rate = 0.0001;
    atomicAdd(m1, learning_rate * error * x1_val);
    atomicAdd(m2, learning_rate * error * x2_val);
    atomicAdd(m3, learning_rate * error * x3_val);
    atomicAdd(c, learning_rate * error);
}

// CUDA Kernel function to perform prediction
__global__ void predict(float* x1, float* x2, float* x3, float* y_pred, float* m1, float* m2, float* m3, float* c) {
    int idx = threadIdx.x;
    float x1_val = x1[idx];
    float x2_val = x2[idx];
    float x3_val = x3[idx];

    y_pred[idx] = (*m1) * x1_val + (*m2) * x2_val + (*m3) * x3_val + (*c);
}

int main() {

    vector<Iris> dataset;

    // Load the dataset from the Iris.csv
    loadDataset(dataset);

    //  Size of dataset
    int N = dataset.size(); // 150

    // Print the preprocessed dataset
    //printPreprocessedDataset(dataset);

    // Set the train and test dataset
    vector<Iris> trainSet, testSet;

    // Initialize random seed
    int random_state = 101;
    srand(random_state);
    float test_size = 0.33;

    for (const Iris& iris : dataset) {
        if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < test_size) {
            testSet.push_back(iris);
        }
        else {
            trainSet.push_back(iris);
        }
    }

    // Print the sizes of the data, training, and testing sets
    cout << "Iris dataset size: " << N << endl;
    cout << "Training set size: " << trainSet.size() << endl;
    cout << "Testing set size: " << testSet.size() << endl;
    cout << "\n\nTraining phase" << endl;

    /* Training */

    // number of samples in the training set
    int train_n = trainSet.size();

    // Size, in bytes, of each vector
    size_t bytes_train = train_n * sizeof(float);

    // Allocate host vectors with training data
    float* h_x1 = (float*)malloc(bytes_train);
    float* h_x2 = (float*)malloc(bytes_train);
    float* h_x3 = (float*)malloc(bytes_train);
    float* h_y = (float*)malloc(bytes_train);

    // Initialize host vectors with training data
    for (int i = 0; i < train_n; i++) {
        h_x1[i] = trainSet[i].sepalWidth;
        h_x2[i] = trainSet[i].petalLength;
        h_x3[i] = trainSet[i].petalWidth;
        h_y[i] = trainSet[i].sepalLength;
    }

    // Allocate device vectors
    float* d_x1, * d_x2, * d_x3, * d_y;
    cudaMalloc(&d_x1, bytes_train);
    cudaMalloc(&d_x2, bytes_train);
    cudaMalloc(&d_x3, bytes_train);
    cudaMalloc(&d_y, bytes_train);

    // Copy data from host to device
    cudaMemcpy(d_x1, h_x1, bytes_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, bytes_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3, h_x3, bytes_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes_train, cudaMemcpyHostToDevice);

    // Allocate device variables for m and c
    float* d_m1, * d_m2, * d_m3, * d_c;
    cudaMalloc(&d_m1, sizeof(float));
    cudaMalloc(&d_m2, sizeof(float));
    cudaMalloc(&d_m3, sizeof(float));
    cudaMalloc(&d_c, sizeof(float));

    // Initialize m and c to 0
    float h_m1 = 0, h_m2 = 0, h_m3 = 0, h_c = 0;
    cudaMemcpy(d_m1, &h_m1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, &h_m2, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m3, &h_m3, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice);

    // Number of iterations for the training phase
    int iterations = 1000;

    // Start recording
    auto start = chrono::high_resolution_clock::now();

    // Call the kernel function for the training phase
    for (int i = 0; i < iterations; i++) {
        linear_regression << < 1, train_n >> > (d_x1, d_x2, d_x3, d_y, d_m1, d_m2, d_m3, d_c);

        cudaDeviceSynchronize();
    }

    // Stop recording
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    // Print time
    cout << fixed << setprecision(6) << "Time to calculate: " << diff.count() << " s" << endl;

    // Copy m and c from device to host
    cudaMemcpy(&h_m1, d_m1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_m2, d_m2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_m3, d_m3, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the learned parameters
    cout << "Learned parameters:" << endl;
    cout << "m1: " << h_m1 << ", m2: " << h_m2 << ", m3: " << h_m3 << ", c: " << h_c << endl;

    cout << "\n\nTesting phase" << endl;

    /* Testing */

    // number of samples in the testing set
    int test_n = testSet.size();

    // Size, in bytes, of each vector
    size_t bytes_test = test_n * sizeof(float);

    // Allocate host vectors with testing data
    float* h_x1_test = (float*)malloc(bytes_test);
    float* h_x2_test = (float*)malloc(bytes_test);
    float* h_x3_test = (float*)malloc(bytes_test);
    float* h_y_test = (float*)malloc(bytes_test);

    // Initialize host vectors with testing data
    for (int i = 0; i < test_n; i++) {
        h_x1_test[i] = testSet[i].sepalWidth;
        h_x2_test[i] = testSet[i].petalLength;
        h_x3_test[i] = testSet[i].petalWidth;
        h_y_test[i] = testSet[i].sepalLength;
    }

    // Allocate device vectors
    float* d_x1_test, * d_x2_test, * d_x3_test, * d_y_pred;
    cudaMalloc(&d_x1_test, bytes_test);
    cudaMalloc(&d_x2_test, bytes_test);
    cudaMalloc(&d_x3_test, bytes_test);
    cudaMalloc(&d_y_pred, bytes_test);

    // Copy data from host to device
    cudaMemcpy(d_x1_test, h_x1_test, bytes_test, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2_test, h_x2_test, bytes_test, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3_test, h_x3_test, bytes_test, cudaMemcpyHostToDevice);

    // Start recording
    start = chrono::high_resolution_clock::now();

    // Call the kernel function for the testing phase
    predict << < 1, test_n >> > (d_x1_test, d_x2_test, d_x3_test, d_y_pred, d_m1, d_m2, d_m3, d_c);
    cudaDeviceSynchronize();

    // Stop recording
    end = chrono::high_resolution_clock::now();
    diff = end - start;

    // Print time
    cout << fixed << setprecision(6) << "Time to calculate: " << diff.count() << " s" << endl;

    // Copy y_pred from device to host
    float* h_y_pred = (float*)malloc(bytes_test);
    cudaMemcpy(h_y_pred, d_y_pred, bytes_test, cudaMemcpyDeviceToHost);

    // Print the mean absolute error, mean squared error, and root mean squared error
    cout << "Mean Absolute Error: " << mean_absolute_error(h_y_test, h_y_pred, test_n) << endl;
    cout << "Mean Squared Error: " << mean_squared_error(h_y_test, h_y_pred, test_n) << endl;
    cout << "Root Mean Squared Error: " << root_mean_squared_error(h_y_test, h_y_pred, test_n) << endl;

    cout << "Predicted Sepal Length(cm): " << h_y_pred[0] << endl;
    cout << "Actual Sepal Length(cm):" << testSet[0].sepalLength << endl;
    cout << "Difference(cm): " << abs(h_y_pred[0] - testSet[0].sepalLength) << endl << endl;

    // Free device memory
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_x3);
    cudaFree(d_y);
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
    cudaFree(d_c);
    cudaFree(d_x1_test);
    cudaFree(d_x2_test);
    cudaFree(d_x3_test);
    cudaFree(d_y_pred);

    // Free host memory
    free(h_x1);
    free(h_x2);
    free(h_x3);
    free(h_y);
    free(h_x1_test);
    free(h_x2_test);
    free(h_x3_test);
    free(h_y_test);
    free(h_y_pred);

    return 0;
}
