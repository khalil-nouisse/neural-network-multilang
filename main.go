package main

import (
	"fmt"
	"go-neural-net/network"
)

func main() {
	// 3 inputs (Req/s, Queue, CPU%), two hidden layers of 4, 1 output node
	nn := network.NewNetwork([]int{3, 4, 4, 1}, 0.1)

	// Training Data (Mock system metrics)
	// Input: {Req/s, Queue, CPU%} | Target: {Spike Probability}
	trainingData := [][][]float64{
		{{0.1, 0.1, 0.2}, {0.0}},
		{{0.8, 0.9, 0.9}, {1.0}},
		{{0.5, 0.5, 0.6}, {0.5}},
		{{0.2, 0.1, 0.1}, {0.0}},
		{{0.9, 0.8, 0.9}, {1.0}},
	}

	epochs := 10000
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			inputs := data[0]
			targets := data[1]
			nn.Train(inputs, targets)
		}
	}

	fmt.Println("Training Complete!")

	// Mock system metrics for prediction: High requests, medium queue, high CPU
	mockMetrics := []float64{0.5, 0.8, 0.8}
	_, prediction := nn.Feedforward(mockMetrics)

	fmt.Printf("Mock Metrics: %v\n", mockMetrics)
	fmt.Printf("Probability of a resource spike: %.2f%%\n", prediction[0]*100)

	// Low metrics
	lowMetrics := []float64{0.1, 0.2, 0.1}
	_, lowPrediction := nn.Feedforward(lowMetrics)
	fmt.Printf("Low Metrics: %v\n", lowMetrics)
	fmt.Printf("Probability of a resource spike: %.2f%%\n", lowPrediction[0]*100)
}
