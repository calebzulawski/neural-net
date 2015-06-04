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

// Constructor for reading network from file
bool Network::loadFromFile(std::string filename) {
	std::ifstream infile(filename, std::ifstream::binary);
	std::string line;
	std::istringstream liness;

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

	weights.resize(2);
	weights[0] = inputWeights;
	weights[1] = outputWeights;
	return true;
}

bool Network::writeToFile(std::string filename) {
	std::ofstream outfile(filename, std::ifstream::binary);
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

	// Edges from input layer to hidden layer
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

void Network::allocateWeights(int _numInputs, int _numHidden, int _numOutput) {
	numInputs = _numInputs;
	numHidden = _numHidden;
	numOutput = _numOutput;

	// Edges from input layer to hidden layer
	inputWeights.resize(numHidden);
	for (unsigned int i = 0; i < inputWeights.size(); i++) {
		inputWeights[i] = std::vector<double>(numInputs+1,0); // +1 for bias
	}

	// Edges from hidden layer to output layer
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

void Network::getActivation(std::vector<double> &_sample, std::vector<double> &_label) {
	// Input layer
	for (int i=0; i < numInputs; i++) {
		activation[0][i] = _sample[i];
	}

	// Hidden and output layer
	for(int l=1; l < 2; l++) {
		for(unsigned int i=0; i < input[l].size(); i++) {
			input[l][i] = -1 * weights[l-1][i][0];
			for(unsigned int j=0; j < input[l-1].size(); j++) {
				input[l][i] += weights[l-1][i][j+1] * activation[l-1][j];
			}
			activation[l][i] = SIGMOID(input[l][i]);
		}
	}
}

void Network::train(Dataset data, double learnRate, int epochs) {
	
}

bool Dataset::loadFromFile(std::string filename) {
	std::ifstream infile(filename, std::ifstream::binary);
	std::string line;
	std::istringstream liness;

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
		std::getline(infile, line);
		liness.str(line);
		liness.clear();
		
		data[i].resize(numFeatures);
		labels[i].resize(numLabels);
		classified[i].resize(numLabels);

		for (int j = 0; j < numFeatures; j++) {
			liness >> data[i][j];
		}
		for (int j = 0; j < numLabels; j++) {
			bool temp;
			liness >> temp;
			labels[i][j] = temp;
		}
	}
	return true;
}

bool Dataset::sample(int i, std::vector<double> &_features, std::vector<bool> &_labels) {
	if (i < numSamples) {
		_features = data[i];
		_labels = labels[i];
		return true;
	} else {
		return false;
	}
}