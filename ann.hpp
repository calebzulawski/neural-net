#ifndef ANN_H
#define ANN_H

#include <vector>

class Network {
public:
	std::vector< std::vector<double>* >* inputWeights;
	std::vector< std::vector<double>* >* outputWeights;
	Network(int numInputs, int numHidden, int numOutput);
};

#endif /* ANN_H */