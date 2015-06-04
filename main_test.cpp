#include <iostream>
#include "ann.hpp"

using namespace std;

int main(int argc, char *argv[]) {
	Network nn;
	nn.loadFromFile(string(argv[1]));

	Dataset testData;
	testData.loadFromFile(string(argv[2]));

	nn.test(testData);

	testData.printStats();
}