CC=g++
CFLAGS=-std=c++11 -c -Wall
LDFLAGS=
SOURCES=main.cpp ann.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=neural-net

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@