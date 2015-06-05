#!/bin/bash

cd ..
if [ -f test ]; then
	TESTPROG=test
	TRAINPROG=train
elif [-f test.exe ]; then
	TESTPROG=test.exe
	TRAINPROG=train.exe
else
	echo 'You must first build the program!'
	exit
fi

echo 'Training from WDBC training set'
$TRAINPROG data/wdbc/sample.NNWDBC.init scripts/wdbc.trained data/wdbc/wdbc.train 0.1 100

echo 'Predicting on WDBC test set'
$TESTPROG scripts/wdbc.trained data/wdbc/wdbc.test scripts/wdbc.results

if [ $(diff data/wdbc/sample.NNWDBC.1.100.trained scripts/wdbc.trained) ]; then
	echo 'The weights are identical!'
else
	echo 'The weights do not match!'
fi

if [ $(diff data/wdbc/sample.NNWDBC.1.100.results scripts/wdbc.results) ]; then
	echo 'The results are identical!'
else
	echo 'The results do not match!'
fi