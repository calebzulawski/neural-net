CC=g++
CFLAGS=-std=c++11 -c -Wall
LDFLAGS=

# Training program
TRAINSOURCES=main_train.cpp ann.cpp
TRAINOBJECTS=$(TRAINSOURCES:.cpp=.o)
TRAINEXE=train

# Testing program
TESTSOURCES=main_test.cpp ann.cpp
TESTOBJECTS=$(TESTSOURCES:.cpp=.o)
TESTEXE=test

all: $(TRAINEXE) $(TESTEXE)

$(TRAINEXE): $(TRAINOBJECTS) 
	$(CC) $(LDFLAGS) $(TRAINOBJECTS) -o $@

$(TESTEXE): $(TESTOBJECTS) 
	$(CC) $(LDFLAGS) $(TESTOBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	@rm -f $(TRAINEXE) $(TESTEXE) *.o