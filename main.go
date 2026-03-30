package main

import (
	"fmt"
	"go-neural-net/network"
)

func main() {
	//3 inputs (Req/s, Queue, CPU%), 4 hidden nodes, 1 output node
	nn := network.NewNetwork(3, 4, 1, 0.1)

	// Mock system metrics: High requests, medium queue, high CPU
	mockMetrics := []float64{1.0, 0.5, 0.8}

	prediction := nn.Feedforward(mockMetrics)

	fmt.Printf("Probability of a resource spike: %.2f%%\n", prediction[0]*100)
}
