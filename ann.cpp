#include "ann.hpp"
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>

#define SIGMOID(x) 1/(1+exp(-x))
#define SIGDERIV(x) SIGMOID(x)*(1-SIGMOID(x))

// Load network weights from `filename`
// Returns true if successful, false if file could not be opened
// Assumes file is in correct form
bool Network::loadFromFile(std::string filename) {
	std::ifstream infile(filename, std::ifstream::binary);
	std::string line;
	std::istringstream liness;

	// Check if file opened
	if (!infile)
		return false;

	// Read header
	std::getline(infile, line);
	liness.str(line);
	liness >> numInputs >> numHidden >> numOutput;

	// Initialize weights
	allocateWeights(numInputs, numHidden, numOutput);

	// Read in edges between input and hidden layer
	for (int i = 0; i < numHidden; i++) {
		std::getline(infile, line);
		liness.str(line);
		liness.clear();
		for (int j = 0; j < numInputs + 1; j++) {
			liness >> inputWeights[i][j];
		}
	}

	// Read in edges between hidden and output layer
	for (int i = 0; i < numOutput; i++) {
		std::getline(infile, line);
		liness.str(line);
		liness.clear();
		for (int j = 0; j < numHidden + 1; j++) {
			liness >> outputWeights[i][j];
		}
	}
	infile.close();

	// Update weights vector
	weights.resize(2);
	weights[0] = inputWeights;
	weights[1] = outputWeights;
	return true;
}

// Writes network weights to `filename`
// Returns true if successful, false if file could not be created
bool Network::writeToFile(std::string filename) {
	std::ofstream outfile(filename, std::ifstream::binary);

	// Check if file created
	if (!outfile)
		return false;

	// Write header
	outfile << numInputs << " " << numHidden << " " << numOutput << std::endl;

	// Edges from input layer to hidden layer
	for (int i = 0; i < numHidden; i++) {
		for (int j = 0; j < numInputs + 1; j++) {
			outfile << std::setiosflags(std::ios::fixed) << std::setprecision(3) << inputWeights[i][j];
			if (j != numInputs)
				outfile << " ";
		}
		outfile << std::endl;
	}

	// Edges from hidden layer to output layer
	for (int i = 0; i < numOutput; i++) {
		for (int j = 0; j < numHidden + 1; j++) {
			outfile << std::setiosflags(std::ios::fixed) << std::setprecision(3) << outputWeights[i][j];
			if (j != numHidden)
				outfile << " ";
		}
		outfile << std::endl;
	}

	outfile.close();
	return true;
}

// Initializes the various weight vectors
void Network::allocateWeights(int _numInputs, int _numHidden, int _numOutput) {
	// Copy to member variables
	numInputs = _numInputs;
	numHidden = _numHidden;
	numOutput = _numOutput;

	// Allocate edges from input layer to hidden layer
	inputWeights.resize(numHidden);
	for (unsigned int i = 0; i < inputWeights.size(); i++) {
		inputWeights[i] = std::vector<double>(numInputs+1,0); // +1 for bias
	}

	// Allocate edges from hidden layer to output layer
	outputWeights.resize(numOutput);
	for (unsigned int i = 0; i < outputWeights.size(); i++) {
		outputWeights[i] = std::vector<double>(numHidden+1,0); // +1 for bias
	}

	// Initialize input and activation vectors
	input.resize(3);
	activation.resize(3);
	input[0].resize(numInputs);
	activation[0].resize(numInputs);
	input[1].resize(numHidden);
	activation[1].resize(numHidden);
	input[2].resize(numOutput);
	activation[2].resize(numOutput);
}

// Calculates the activation of each neuron for a particular input `_sample`
void Network::getActivation(std::vector<double> &_sample) {
	// Input layer activations equal to input sample
	for (int i=0; i < numInputs; i++) {
		activation[0][i] = _sample[i];
	}

	// Solve hidden and output layer activations
	for(int l=1; l < 3; l++) {
		for(unsigned int i=0; i < input[l].size(); i++) {
			input[l][i] = -1 * weights[l-1][i][0];
			for(unsigned int j=0; j < input[l-1].size(); j++) {
				input[l][i] += weights[l-1][i][j+1] * activation[l-1][j];
			}
			activation[l][i] = SIGMOID(input[l][i]);
		}
	}
}

// Display all activations for debugging
void Network::printActivation() {
	for (int l=0; l < 3; l++) {
		std::cout << "Layer " << l << std::endl;
		for (unsigned int i=0; i < activation[l].size(); i++) {
			std::cout << activation[l][i] << " ";
		}
		std::cout << std::endl;
	}
}

// Train network on dataset `data` with learning rate `learnRate` for `epochs` epochs
void Network::train(Dataset &data, double learnRate, int epochs) {
	// Loop over epochs
	for (int iter=0; iter < epochs; iter++) {
		// Loop over samples
		for (int i=0; i < data.size(); i++) {
			std::vector<double> sampleFeatures;
			std::vector<bool> sampleLabels;
			std::vector<double> outputDelta(numOutput,0);
			std::vector<double> hiddenDelta(numHidden,0);

			// Read sample and calculate activations
			data.sample(i, sampleFeatures, sampleLabels);
			getActivation(sampleFeatures);

			// Output layer delta
			for (int j=0; j < numOutput; j++) {
				outputDelta[j] = SIGDERIV(input[2][j]) * (sampleLabels[j] - activation[2][j]);
			}

			// Hidden layer delta
			for (int j=0; j < numHidden; j++) {
				for (int k=0; k < numOutput; k++) {
					hiddenDelta[j] += outputWeights[k][j+1] * outputDelta[k];
				}
				hiddenDelta[j] *= SIGDERIV(input[1][j]);
			}

			// Update weights from hidden layer to output layer
			for (int j=0; j < numOutput; j++) {
				for (int k=0; k < numHidden; k++) {
					outputWeights[j][k+1] = outputWeights[j][k+1] + learnRate * activation[1][k] * outputDelta[j];
				}
				outputWeights[j][0] = outputWeights[j][0] + learnRate * -1 * outputDelta[j]; // Bias
			}

			// Update weights from input layer to hidden layer
			for (int j=0; j < numHidden; j++) {
				for (int k=0; k < numInputs; k++) {
					inputWeights[j][k+1] = inputWeights[j][k+1] + learnRate * activation[0][k] * hiddenDelta[j];
				}
				inputWeights[j][0] = inputWeights[j][0] + learnRate * -1 * hiddenDelta[j]; // Bias
			}

			// Update weight vector
			weights[0] = inputWeights;
			weights[1] = outputWeights;
		}
	}
}

// Predict on `data` with current network weights
void Network::test(Dataset &data) {
	// Loop over samples
	for (int i=0; i < data.size(); i++) {
		std::vector<double> sampleFeatures;
		std::vector<bool> sampleLabels;

		// Load current sample and calculate activation
		data.sample(i, sampleFeatures, sampleLabels);
		getActivation(sampleFeatures);

		// Loop over each class
		for (int j=0; j < numOutput; j++) {
			// Threshold activations to predict
			data.classify(i, j, (activation[2][j] >= .5 ? true : false));
		}
	}
}

// Load a dataset from `filename`
// Returns true if successful, false if could not open file
// Assumes file is in correct form
bool Dataset::loadFromFile(std::string filename) {
	std::ifstream infile(filename, std::ifstream::binary);
	std::string line;
	std::istringstream liness;

	// Check if file was opened
	if (!infile)
		return false;

	// Read header
	std::getline(infile, line);
	liness.str(line);
	liness >> numSamples >> numFeatures >> numLabels;

	// Read samples
	data.resize(numSamples);
	labels.resize(numSamples);
	classified.resize(numSamples);
	for (int i = 0; i < numSamples; i++) {
		// Read line into stringstream and reset pointers
		std::getline(infile, line);
		liness.str(line);
		liness.clear();
		
		// Resize subvectors
		data[i].resize(numFeatures);
		labels[i].resize(numLabels);
		classified[i].resize(numLabels);

		// Extract values from line stringstream
		for (int j = 0; j < numFeatures; j++) {
			liness >> data[i][j];
		}
		for (int j = 0; j < numLabels; j++) {
			// labels[i][j] returns a reference object, not a bool
			// This is due to efficient implementation of std::vector<bool>
			// Does not allow extraction operator >> to work in its current form
			// bool temp is a simple workaround
			bool temp;
			liness >> temp;
			labels[i][j] = temp;
		}
	}
	return true;
}

// Copy the `i`th sample into `_features` and `_labels`
// Returns true if the sample exists, false if `i` exceeds bounds
bool Dataset::sample(int i, std::vector<double> &_features, std::vector<bool> &_labels) {
	if (i < numSamples) {
		_features = data[i];
		_labels = labels[i];
		return true;
	} else {
		return false;
	}
}

// Copy prediction for the `i`th sample and `j`th class into dataset
// Returns true if successful, false if `i` or `j` exceeds bounds
bool Dataset::classify(int i, int j, bool prediction) {
	if ((i < numSamples) & (j < numLabels)) {
		classified[i][j] = prediction;
		return true;
	} else {
		return false;
	}
}

// Calculate various metrics for the prediction and writes to `filename`
// Returns true if successful, false if file could not be created
bool Dataset::writeStatsToFile(std::string filename) {
	std::ofstream outfile(filename, std::ifstream::binary);

	// Check if file created
	if (!outfile)
		return false;

	// Accumulators
	double totalA, totalB, totalC, totalD;

	// Overall metrics
	double microAccuracy, microPrecision, microRecall, microF1;
	double macroAccuracy, macroPrecision, macroRecall, macroF1;

	// Loop over all classes
	for (int l=0; l < numLabels; l++) {
		// Reset counts
		double A=0, B=0, C=0, D=0;
		// Loop over samples and update counts
		for (int i=0; i < numSamples; i++) {
			if (labels[i][l] && classified[i][l])
				A++;
			if (!labels[i][l] && classified[i][l])
				B++;
			if (labels[i][l] && !classified[i][l])
				C++;
			if (!labels[i][l] && !classified[i][l])
				D++;
		}

		// Calculate class metrics
		double accuracy = (A + D) / (A + B + C + D);
		double precision = A / (A + B);
		double recall = A / (A + C);
		double f1 = (2 * precision * recall) / (precision + recall);

		// Write counts and metrics to file
		outfile << std::setiosflags(std::ios::fixed) << std::setprecision(0) << A << " " << B << " " << C << " " << D << " " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accuracy << " " << precision << " " << recall << " " << f1 << std::endl;

		// Update accumulated counts
		totalA += A;
		totalB += B;
		totalC += C;
		totalD += D;

		// Update macro averages
		macroAccuracy += (accuracy/numLabels);
		macroPrecision += (precision/numLabels);
		macroRecall += (recall/numLabels);
	}

	// Calculate micro averages
	microAccuracy = (totalA + totalD) / (totalA + totalB + totalC + totalD);
	microPrecision = totalA / (totalA + totalB);
	microRecall = totalA / (totalA + totalC);
	microF1 = (2 * microPrecision * microRecall) / (microPrecision + microRecall);

	// Calculate macro F1
	macroF1 = (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall);

	// Write average metrics to file
	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(3) << microAccuracy << " " << microPrecision << " " << microRecall << " " << microF1 << std::endl;
	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(3) << macroAccuracy << " " << macroPrecision << " " << macroRecall << " " << macroF1 << std::endl;

	outfile.close();
	return true;
}