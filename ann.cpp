#include "ann.hpp"
#include <cstddef>

Network::Network(int numInputs, int numHidden, int numOutput) {
	// Edges from input layer to hidden layer
	inputWeights = new std::vector< std::vector<double>* >(numHidden, nullptr);
	for (unsigned int i = 0; i < inputWeights->size(); i++) {
		(*inputWeights)[i] = new std::vector<double>(numInputs+1, 0); // +1 for bias weight
	}

	// Edges from hidden layer to output layer
	outputWeights = new std::vector< std::vector<double>* >(numOutput, nullptr);
	for (unsigned int i = 0; i < outputWeights->size(); i++) {
		(*inputWeights)[i] = new std::vector<double>(numHidden+1, 0); // +1 for bias weight
	}
}