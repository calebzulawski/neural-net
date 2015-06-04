#ifndef ANN_H
#define ANN_H

#include <vector>
#include <string>

class Dataset {
private:
	std::vector< std::vector<double> > data;
	std::vector< std::vector<bool> > labels;
	std::vector< std::vector<bool> > classified;
	int numSamples, numFeatures, numLabels;
public:
	bool loadFromFile(std::string filename);
	bool sample(int i, std::vector<double> &_features, std::vector<bool> &_labels);
};

class Network {
private:
	int numInputs, numHidden, numOutput;
	std::vector< std::vector<double> > inputWeights;
	std::vector< std::vector<double> > outputWeights;
	std::vector< std::vector< std::vector<double> > > weights;
	std::vector< std::vector<double> > input;
	std::vector< std::vector<double> > activation;
public:
	void allocateWeights(int _numInputs, int _numHidden, int _numOutput);
	bool loadFromFile(std::string filename);
	bool writeToFile(std::string filename);
	void getActivation(std::vector<double> &_sample, std::vector<double> &_label);
	void train(Dataset data, double learnRate, int epochs);
};

#endif /* ANN_H */