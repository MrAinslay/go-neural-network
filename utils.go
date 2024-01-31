package main

import "os"

func save(net Network) {
	h, err := os.Create("data/hweights.model")
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	defer h.Close()

	o, err := os.Create("data/oweights.model")
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
	defer o.Close()
}

func load(net *Network) {
	h, err := os.Open("data/hweights.model")
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	defer h.Close()

	o, err := os.Open("data/oweights.model")
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	defer o.Close()
}
