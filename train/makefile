
all: compile

# ommiting: -g

compile:
	icc -std=c++0x -o train SynapseGroup.cpp LIFGroup.cpp load_mnist.cpp train.cpp -Wfatal-errors -Ofast -lpthread

run:
	./train

