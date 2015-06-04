#include <iostream>
#include "ann.hpp"

using namespace std;

int main(int argc, char *argv[]) {
	Network nn;
	nn.loadFromFile("data/sample.NNWDBC.init");

	Dataset trainData;
	trainData.loadFromFile("data/wdbc.train");

	nn.train(trainData, 0.1, 100);

	nn.writeToFile("nn.trained");
}