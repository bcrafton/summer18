
all: compile

# ommiting: -g

compile:
	g++ -std=c++0x -o train SynapseGroup.cpp LIFGroup.cpp load_mnist.cpp train.cpp -Wfatal-errors -Ofast

run:
	./train

