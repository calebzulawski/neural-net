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

void Network::getActivation(std::vector<double> &_sample, std::vector<bool> &_label) {
	// Input layer
	for (int i=0; i < numInputs; i++) {
		activation[0][i] = _sample[i];
	}

	// Hidden and output layer
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

void Network::printActivation() {
	for (int l=0; l < 3; l++) {
		std::cout << "Layer " << l << std::endl;
		for (unsigned int i=0; i < activation[l].size(); i++) {
			std::cout << activation[l][i] << " ";
		}
		std::cout << std::endl;
	}
}

void Network::train(Dataset &data, double learnRate, int epochs) {
	for (int iter=0; iter < epochs; iter++) {
		for (int i=0; i < data.size(); i++) {
			std::vector<double> sampleFeatures;
			std::vector<bool> sampleLabels;
			std::vector<double> outputDelta(numOutput,0);
			std::vector<double> hiddenDelta(numHidden,0);

			data.sample(i, sampleFeatures, sampleLabels);
			getActivation(sampleFeatures, sampleLabels);

			// Output layer
			for (int j=0; j < numOutput; j++) {
				outputDelta[j] = SIGDERIV(input[2][j]) * (sampleLabels[j] - activation[2][j]);
			}

			// Hidden layer
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

void Network::test(Dataset &data) {
	for (int i=0; i < data.size(); i++) {
		std::vector<double> sampleFeatures;
		std::vector<bool> sampleLabels;
		data.sample(i, sampleFeatures, sampleLabels);
		getActivation(sampleFeatures, sampleLabels);
		std::vector<bool> classifiedVec(numOutput);
		for (int j=0; j < numOutput; j++) {
			data.classify(i, j, (activation[2][j] >= .5 ? true : false));
		}
	}
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

bool Dataset::classify(int i, int j, bool prediction) {
	if ((i < numSamples) & (j < numLabels)) {
		classified[i][j] = prediction;
		return true;
	} else {
		return false;
	}
}

bool Dataset::writeStatsToFile(std::string filename) {
	std::ofstream outfile(filename, std::ifstream::binary);
	if (!outfile)
		return false;

	double totalA, totalB, totalC, totalD;
	double microAccuracy, microPrecision, microRecall, microF1;
	double macroAccuracy, macroPrecision, macroRecall, macroF1;
	for (int l=0; l < numLabels; l++) {
		double A=0, B=0, C=0, D=0;
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
		double accuracy = (A + D) / (A + B + C + D);
		double precision = A / (A + B);
		double recall = A / (A + C);
		double f1 = (2 * precision * recall) / (precision + recall);
		outfile << std::setiosflags(std::ios::fixed) << std::setprecision(0) << A << " " << B << " " << C << " " << D << " " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accuracy << " " << precision << " " << recall << " " << f1 << std::endl;

		totalA += A;
		totalB += B;
		totalC += C;
		totalD += D;

		macroAccuracy += (accuracy/numLabels);
		macroPrecision += (precision/numLabels);
		macroRecall += (recall/numLabels);
	}

	microAccuracy = (totalA + totalD) / (totalA + totalB + totalC + totalD);
	microPrecision = totalA / (totalA + totalB);
	microRecall = totalA / (totalA + totalC);
	microF1 = (2 * microPrecision * microRecall) / (microPrecision + microRecall);

	macroF1 = (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall);

	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(3) << microAccuracy << " " << microPrecision << " " << microRecall << " " << microF1 << std::endl;
	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(3) << macroAccuracy << " " << macroPrecision << " " << macroRecall << " " << macroF1 << std::endl;

	outfile.close();
	return true;
}