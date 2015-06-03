#include <iostream>
#include <vector>
#include "ann.hpp"

using namespace std;

int main(int argc, char *argv[]) {
	Network nn;
	nn.loadFromFile("data/sample.NNWDBC.init");
	nn.writeToFile("nn.trained");
}