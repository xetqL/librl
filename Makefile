CC=g++
STD=c++11

IDIR=include/

LOCAL_INSTALLED=/usr/local/

IFLAGS=-I$(IDIR) -I$(LOCAL_INSTALLED)/include

LFLAGS=-L$(LOCAL_INSTALLED)/lib

lFLAGS=-ldoublefann

CPPFLAGS=-Wall -std=$(STD)

examples: gridworld tictactoe

gridworld: examples/gridworld/*.cpp
	$(CC)  $^ -o bin/$@ $(CPPFLAGS) $(IFLAGS) $(LFLAGS) $(lFLAGS)

tictactoe: examples/tictactoe/*.cpp
	$(CC)  $^ -o bin/$@ $(CPPFLAGS) $(IFLAGS) $(LFLAGS) $(lFLAGS)
	
.PHONY: clean
clean:
	@rm -f bin/*

