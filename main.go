package main

import "flag"

func main() {
	/*
		784 inputs - 28 x 28 pixels, each pixel is an input
		200 hidden neurons - an arbitrary number
		10 outputs - digits 0 to 9
		0.1 is the learning rate
	*/

	net := CreateNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	flag.Parse()

	// train os mass predict to determine the effectiveness of the trained network
	switch *mnist {
	case "train":
		mnistTrain(&net)
		save(net)
	case "predict":
		load(&net)
		mnistPredict(&net)
	default:
		//do nothing
	}
}
