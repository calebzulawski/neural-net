#include <iostream>
#include "ann.hpp"

using namespace std;

int main(int argc, char *argv[]) {
	string netFilename, dataFilename, resultsFilename;
	
	// ----------------------------
	// READ INPUTS FROM ARGV OR CIN
	// ----------------------------
	if (argc == 0) {
		// TODO: Type checks on learning rate/epochs
		cout << "Trained weights filename: ";
		cin >> netFilename;
		cout << "Test data filename: ";
		cin >> dataFilename;
		cout << "Prediction results filename: ";
		cin >> resultsFilename;
	}
	else if (argc == 4) {
		netFilename = argv[1];
		dataFilename = argv[2];
		resultsFilename = argv[3];
	}
	else {
		cout << "Invalid options!" << endl << "Usage: test trainedWeightsFile testDataFile resultsOutputFile" << endl;
		return 0;
	}

	// ------------------------------
	// LOAD NETWORK WEIGHTS FROM FILE
	// ------------------------------
	Network nn;
	if (nn.loadFromFile(netFilename)) {
		cout << "Loaded trained weights from '" << netFilename << "'..." << endl;
	} else {
		cout << "Could not open trained weights file '" << netFilename << "'!" << endl;
		return 0;
	}

	// ------------------------
	// LOAD TEST DATA FROM FILE
	// ------------------------
	Dataset testData;
	if (testData.loadFromFile(dataFilename)) {
		cout << "Loaded test data from '" << dataFilename << "'..." << endl;
	} else {
		cout << "Could not open test data file '" << dataFilename << "'!" << endl;
		return 0;
	}
 
	// ------------------------
	// PREDICT TEST DATA LABELS
	// ------------------------
	cout << "Predicting labels for test data..." << endl;
	nn.test(testData);

	// ---------------------
	// WRITE RESULTS TO FILE
	// ---------------------
	if (testData.writeStatsToFile(resultsFilename)) {
		cout << "Wrote results to '" << resultsFilename << "'..." << endl;
	} else {
		cout << "Could not create results file '" << resultsFilename << "'!" << endl;
		return 0;
	}
}