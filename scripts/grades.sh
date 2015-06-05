#!/bin/bash
INIT=data/grades/sample.NNGrades.init
SAMPLEW=data/grades/sample.NNGrades.05.100.trained
TRAINEDW=scripts/grades.trained
TRAINDATA=data/grades/grades.train
TESTDATA=data/grades/grades.test
LEARNRATE=0.05
EPOCHS=100
MYRESULTS=scripts/grades.results
SAMPLERESULTS=data/grades/sample.NNGrades.05.100.results

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
./$TRAINPROG $INIT $TRAINEDW $TRAINDATA $LEARNRATE $EPOCHS

echo 'Predicting on WDBC test set'
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