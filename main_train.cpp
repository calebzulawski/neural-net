#include <iostream>
#include <sstream>
#include "ann.hpp"

using namespace std;

int main(int argc, char *argv[]) {
	int epochs;
	double learnRate;
	string initFilename, trainFilename, dataFilename;

	// ----------------------------
	// READ INPUTS FROM ARGV OR CIN
	// ----------------------------
	if (argc == 0) {
		// TODO: Type checks on learning rate/epochs
		cout << "Initial weights filename: ";
		cin >> initFilename;
		cout << "Trained weights filename: ";
		cin >> trainFilename;
		cout << "Training data filename: ";
		cin >> dataFilename;
		cout << "Learning rate: ";
		cin >> learnRate;
		cout << "Number of epochs: ";
		cin >> epochs;
	}
	else if (argc == 6) {
		initFilename = argv[1];
		trainFilename = argv[2];
		dataFilename = argv[3];

		istringstream ss1(argv[4]);
		// Check if supplied learning rate is numeric
		// Use tellg() since >> will cast until it hits a non-numeric successfully
		if (!(ss1 >> learnRate) || (ss1.tellg() != -1)) {
			cout << "The learning rate " << argv[4] << " is not a valid floating point number!" << endl;
			return 0;
		}

		istringstream ss2(argv[5]);
		// Same check for epochs (an int)
		if (!(ss2 >> epochs) || (ss2.tellg() != -1)) {
			cout << "The number of epochs " << argv[5] << " is not a valid integer!" << endl;
			return 0;
		}
	}
	else {
		cout << "Invalid options!" << endl << "Usage: train initWeightsFile outputWeightsFile trainDataFile learnRate epochs" << endl;
		return 0;
	}

	// ------------------------------
	// LOAD NETWORK WEIGHTS FROM FILE
	// ------------------------------
	Network nn;
	if (nn.loadFromFile(initFilename)) {
		cout << "Loaded initial weights from '" << initFilename << "'..." << endl;
	} else {
		cout << "Could not open initial weights file '" << initFilename << "'!" << endl;
		return 0;
	}


	// ----------------------------
	// LOAD TRAINING DATA FROM FILE
	// ----------------------------
	Dataset trainData;
	if (trainData.loadFromFile(dataFilename)) {
		cout << "Loaded training data from '" << dataFilename << "'..." << endl;
	} else {
		cout << "Could not open training data file '" << dataFilename << "'!" << endl;
		return 0;
	}

	// -------------
	// TRAIN NETWORK
	// -------------
	cout << "Training using a learning rate of " << learnRate << " and " << epochs << " epochs..." << endl;
	nn.train(trainData, learnRate, epochs);

	// -------------------------------------
	// WRITE TRAINED NETWORK WEIGHTS TO FILE
	// -------------------------------------
	if (nn.writeToFile(trainFilename)) {
		cout << "Wrote trained weights to '" << trainFilename << "'..." << endl;
	} else {
		cout << "Could not create weights file '" << trainFilename << "'!" << endl;
		return 0;
	}
}