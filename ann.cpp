#include "ann.hpp"
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

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
}