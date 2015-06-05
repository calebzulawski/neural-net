#ifndef ANN_H
#define ANN_H

#include <vector>
#include <string>

// Class for storing and analyzing data
class Dataset {
private:
	// Features are in the form data[sample][feature]
	std::vector< std::vector<double> > data;
	// Given labels in the form labels[sample][class]
	std::vector< std::vector<bool> > labels;
	// Predicted labels i n the form classified[sample][class]
	std::vector< std::vector<bool> > classified;
	// Number of samples, features, and classes
	int numSamples, numFeatures, numLabels;
public:
	// Method to load a dataset from a file
	bool loadFromFile(std::string filename);
	// Method to copy the ith sample to given _features and _labels vectors
	bool sample(int i, std::vector<double> &_features, std::vector<bool> &_labels);
	// Method to update the prediction of the jth class of the ith sample
	bool classify(int i, int j, bool prediction);
	// Number of samples get method
	int size(){ return numSamples; };
	// Method to calculate various metrics and write to file
	bool writeStatsToFile(std::string filename);
};

// Class for the artificial neural network
class Network {
private:
	// Number of neurons in input, hidden, and output layers
	int numInputs, numHidden, numOutput;
	// Weights from ith input neuron to jth hidden neuron in form inputWeights[j][i]
	std::vector< std::vector<double> > inputWeights;
	// Weights from ith hidden neuron to jth output neuron in form outputWeights[j][i]
	std::vector< std::vector<double> > outputWeights;
	// All weights combined into one vector in the form weights[layer][j][i] where input layer is 0
	std::vector< std::vector< std::vector<double> > > weights;
	// Vector for input in the form input[layer][neuron]
	std::vector< std::vector<double> > input;
	// Vector for activation in the form activation[layer][neuron]
	std::vector< std::vector<double> > activation;
public:
	// Method for setting up all 0 weights when initializing the network
	void allocateWeights(int _numInputs, int _numHidden, int _numOutput);
	// Method to load weights from a file
	bool loadFromFile(std::string filename);
	// Method to write weights to a file
	bool writeToFile(std::string filename);
	// Method to evaluate activations for a particular input
	void getActivation(std::vector<double> &_sample);
	// Print the activations to cout for debugging
	void printActivation();
	// Train the network from a dataset
	void train(Dataset &data, double learnRate, int epochs);
	// Predict on a dataset
	void test(Dataset &data);
};

#endif /* ANN_H */