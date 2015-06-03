#ifndef ANN_H
#define ANN_H

#include <vector>
#include <string>

class Network {
public:
	int numInputs, numHidden, numOutput;
	std::vector< std::vector<double> > inputWeights;
	std::vector< std::vector<double> > outputWeights;
	void allocateWeights(int _numInputs, int _numHidden, int _numOutput);
	bool loadFromFile(std::string filename);
	bool writeToFile(std::string filename);
};

#endif /* ANN_H */