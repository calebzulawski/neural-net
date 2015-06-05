#!/bin/bash
INIT=data/wdbc/sample.NNWDBC.init
SAMPLEW=data/wdbc/sample.NNWDBC.1.100.trained
TRAINEDW=scripts/wdbc.trained
TRAINDATA=data/wdbc/wdbc.train
TESTDATA=data/wdbc/wdbc.test
LEARNRATE=0.1
EPOCHS=100
MYRESULTS=scripts/wdbc.results
SAMPLERESULTS=data/wdbc/sample.NNWDBC.1.100.results

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

echo 'Training from training set'
./$TRAINPROG $INIT $TRAINEDW $TRAINDATA $LEARNRATE $EPOCHS

echo 'Predicting on test set'
./$TESTPROG $TRAINEDW $TESTDATA $MYRESULTS

echo
if diff -q $SAMPLEW $TRAINEDW > /dev/null; then
	echo 'The weights are identical!'
else
	echo 'The weights do not match!'
fi

if diff -q $SAMPLERESULTS $MYRESULTS > /dev/null; then
	echo 'The results are identical!'
else
	echo 'The results do not match!'
fi