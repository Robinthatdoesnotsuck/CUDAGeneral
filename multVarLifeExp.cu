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

struct LifeExp {

    int Country;
    int Year;
    int Status;
    float Life;
    float Adult;
    float Infant;
    float Alcohol;
    float Percentage_E;
    float Hepatitis_B;
    float Measles;
    float BMI;
    float U5_Deaths;
    float Polio;
    float Tot_E;
    float Diphtheria;
    float AIDS;
    float GDP;
    float Population;
    float Thinness_19;
    float Thinness_59;
    float Income;
    float Schooling;
};

void eraseFileLine(string path, string eraseLine) {
    string line;
    ifstream fin;

    fin.open(path);
    // contents of path must be copied to a temp file then
    // renamed back to the path file
    ofstream temp;
    temp.open("temp.txt");

    while (getline(fin, line)) {
        // write all lines to temp other than the line marked for erasing
        if (line != eraseLine)
            temp << line << endl;
    }

    temp.close();
    fin.close();

    // required conversion for remove and rename functions
    const char* p = path.c_str();
    remove(p);
    rename("temp.txt", p);
}

void loadDataset(vector<LifeExp>& d) {
    //ifstream file("LifeExpectancyData.csv");
    ifstream file("LifeExpectancyData.csv/LifeExpectancyData.csv");
    string line;
    string rowS;
    int rowI = 0;
    map<string, int> countryMap;
    map<string, int> statusMap;
    int countryCounter = 0;
    int statusCounter = 0;
    // Skip header line
    getline(file, line);

    while (getline(file, line)) {
        bool missing = false;
        stringstream ss(line);
        string cell;

        LifeExp lifeExp;

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }

        if (countryMap.find(cell) == countryMap.end()) {
            countryMap[cell] = countryCounter++;
        }
        lifeExp.Country = countryMap[cell];

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Year = stoi(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        if (statusMap.find(cell) == statusMap.end()) {
            statusMap[cell] = statusCounter++;
        }
        lifeExp.Status = statusMap[cell];

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Life = stof(cell);


        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Adult = stof(cell);


        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Infant = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }

        lifeExp.Alcohol = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            missing = true;
            cell = "0";
        }
        lifeExp.Percentage_E = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Hepatitis_B = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Measles = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.BMI = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.U5_Deaths = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Polio = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Tot_E = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Diphtheria = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.AIDS = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.GDP = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Population = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Thinness_19 = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            cell = "0";
        }
        lifeExp.Thinness_59 = stof(cell);

        getline(ss, cell, ',');
        if (cell == "")
        {
            missing = true;
            cell = "0";
        }
        lifeExp.Income = stof(cell);

        getline(ss, cell, ',');
        if (cell == "" || cell == "\r")
        {
            missing = true;
            cell = "0";
        }
        lifeExp.Schooling = stof(cell);
        if(!missing) {
            d.push_back(lifeExp);
        }

    }
}

void printPreprocessedDataset(vector<LifeExp>& d) {
    for (const LifeExp& lifeExp : d) {

        cout << ", \tCountry: " << lifeExp.Country
            << ", \tYear: " << lifeExp.Year
            << ", \tLife: " << lifeExp.Life
            << ", \tAdult: " << lifeExp.Adult
            << ", \tInfant: " << lifeExp.Infant
            << ", \tAlcohol: " << lifeExp.Alcohol
            << ", \tPercentage_E: " << lifeExp.Percentage_E
            << ", \tHepatitis_B: " << lifeExp.Hepatitis_B
            << ", \tMeasles: " << lifeExp.Measles
            << ", \tBMI: " << lifeExp.BMI
            << ", \tU5_Deaths: " << lifeExp.U5_Deaths
            << ", \tPolio: " << lifeExp.Polio
            << ", \tTot_E: " << lifeExp.Tot_E
            << ", \tDiphtheria: " << lifeExp.Diphtheria
            << ", \tAIDS: " << lifeExp.AIDS
            << ", \tGDP: " << lifeExp.GDP
            << ", \tPopulation: " << lifeExp.Population
            << ", \tThinness_19: " << lifeExp.Thinness_19
            << ", \tThinness_59: " << lifeExp.Thinness_59
            << ", \tIncome: " << lifeExp.Income
            << ", \tSchooling: " << lifeExp.Schooling << endl;
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
__global__ void linear_regression(float* x1, float* x2, float* y, float* m1, float* m2, float* c) {
    int idx = threadIdx.x;
    float x1_val = x1[idx];
    float x2_val = x2[idx];
    //float x3_val = x3[idx];
    float y_val = y[idx];

    float prediction = (*m1) * x1_val + (*m2) * x2_val + (*c);
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
    //atomicAdd(m3, learning_rate * error * x3_val);
    atomicAdd(c, learning_rate * error);
}

// CUDA Kernel function to perform prediction
__global__ void predict(float* x1, float* x2, float* y_pred, float* m1, float* m2, float* c) {
    int idx = threadIdx.x;
    float x1_val = x1[idx];
    float x2_val = x2[idx];
    //float x3_val = x3[idx];

    y_pred[idx] = (*m1) * x1_val + (*m2) * x2_val + (*c);
}

int main() {

    vector<LifeExp> dataset;

    // Load the dataset from the LifeExp.csv
    loadDataset(dataset);

    //  Size of dataset
    int N = dataset.size(); // 150

    // Print the preprocessed dataset
    //printPreprocessedDataset(dataset);

    // Set the train and test dataset
    vector<LifeExp> trainSet, testSet;

    // Initialize random seed
    int random_state = 101;
    srand(random_state);
    float test_size = 0.33;

    for (const LifeExp& LifeExp : dataset) {
        if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < test_size) {
            testSet.push_back(LifeExp);
        }
        else {
            trainSet.push_back(LifeExp);
        }
    }

    // Print the sizes of the data, training, and testing sets
    cout << "LifeExp dataset size: " << N << endl;
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
    float* h_y = (float*)malloc(bytes_train);

    // Initialize host vectors with training data
    for (int i = 0; i < train_n; i++) {
        h_x1[i] = trainSet[i].Schooling;
        h_x2[i] = trainSet[i].Income;
        h_y[i] = trainSet[i].Percentage_E;
    }

    // Allocate device vectors
    float* d_x1, * d_x2, * d_y;
    cudaMalloc(&d_x1, bytes_train);
    cudaMalloc(&d_x2, bytes_train);
    //cudaMalloc(&d_x3, bytes_train);
    cudaMalloc(&d_y, bytes_train);

    // Copy data from host to device
    cudaMemcpy(d_x1, h_x1, bytes_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, bytes_train, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_x3, h_x3, bytes_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes_train, cudaMemcpyHostToDevice);

    // Allocate device variables for m and c
    float* d_m1, * d_m2, * d_c;
    cudaMalloc(&d_m1, sizeof(float));
    cudaMalloc(&d_m2, sizeof(float));
    //cudaMalloc(&d_m3, sizeof(float));
    cudaMalloc(&d_c, sizeof(float));

    // Initialize m and c to 0
    float h_m1 = 0, h_m2 = 0, h_c = 0;
    cudaMemcpy(d_m1, &h_m1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, &h_m2, sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_m3, &h_m3, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice);

    // Number of iterations for the training phase
    int iterations = 1000;

    // Start recording
    auto start = chrono::high_resolution_clock::now();

    // Call the kernel function for the training phase
    for (int i = 0; i < iterations; i++) {
        linear_regression << < 1, train_n >> > (d_x1, d_x2, d_y, d_m1, d_m2, d_c);
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
    //cudaMemcpy(&h_m3, d_m3, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the learned parameters
    cout << "Learned parameters:" << endl;
    cout << "m1: " << h_m1 << ", m2: " << h_m2 << ", c: " << h_c << endl;
    //cout << "m1: " << h_m1 << ", c: " << h_c << endl;

    cout << "\n\nTesting phase" << endl;

    /* Testing */

    // number of samples in the testing set
    int test_n = testSet.size();

    // Size, in bytes, of each vector
    size_t bytes_test = test_n * sizeof(float);

    // Allocate host vectors with testing data
    float* h_x1_test = (float*)malloc(bytes_test);
    float* h_x2_test = (float*)malloc(bytes_test);
    //float* h_x3_test = (float*)malloc(bytes_test);
    float* h_y_test = (float*)malloc(bytes_test);

    // Initialize host vectors with testing data
    for (int i = 0; i < test_n; i++) {
        h_x1_test[i] = testSet[i].GDP;
        h_x2_test[i] = testSet[i].Income;
        //h_x3_test[i] = testSet[i].petalWidth;
        h_y_test[i] = testSet[i].Percentage_E;
    }

    // Allocate device vectors
    float* d_x1_test, * d_x2_test, * d_y_pred;
    cudaMalloc(&d_x1_test, bytes_test);
    cudaMalloc(&d_x2_test, bytes_test);
    //cudaMalloc(&d_x3_test, bytes_test);
    cudaMalloc(&d_y_pred, bytes_test);

    // Copy data from host to device
    cudaMemcpy(d_x1_test, h_x1_test, bytes_test, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2_test, h_x2_test, bytes_test, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_x3_test, h_x3_test, bytes_test, cudaMemcpyHostToDevice);

    // Start recording
    start = chrono::high_resolution_clock::now();

    // Call the kernel function for the testing phase
    predict << < 1, test_n >> > (d_x1_test, d_x2_test, d_y_pred, d_m1, d_m2, d_c);
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

    cout << "Predicted Percentage Expenditure: " << h_y_pred[0] << endl;
    cout << "Actual Percentage Expenditure:" << testSet[0].Percentage_E << endl;
    cout << "Difference: " << abs(h_y_pred[0] - testSet[0].Percentage_E) << endl << endl;

    // Free device memory
    cudaFree(d_x1);
    cudaFree(d_x2);
    //cudaFree(d_x3);
    cudaFree(d_y);
    cudaFree(d_m1);
    cudaFree(d_m2);
    //cudaFree(d_m3);
    cudaFree(d_c);
    cudaFree(d_x1_test);
    cudaFree(d_x2_test);
    //cudaFree(d_x3_test);
    cudaFree(d_y_pred);

    // Free host memory
    free(h_x1);
    free(h_x2);
    //free(h_x3);
    free(h_y);
    free(h_x1_test);
    free(h_x2_test);
    //free(h_x3_test);
    free(h_y_test);
    free(h_y_pred);

    return 0;
}
