package network

import (
	"math/rand"
	"time"
)

type NeuralNetwork struct {
	LayerSizes []int

	// Weights[l] are weights from layer l to l+1
	Weights [][][]float64

	// Biases[l] are biases for layer l+1
	Biases [][][]float64

	LearningRate float64
}

func NewNetwork(layerSizes []int, learningRate float64) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	numLayers := len(layerSizes)

	weights := make([][][]float64, numLayers-1)
	biases := make([][][]float64, numLayers-1)

	for i := 0; i < numLayers-1; i++ {
		weights[i] = randomMatrix(layerSizes[i], layerSizes[i+1])
		biases[i] = randomMatrix(layerSizes[i+1], 1)
	}

	return &NeuralNetwork{
		LayerSizes:   layerSizes,
		Weights:      weights,
		Biases:       biases,
		LearningRate: learningRate,
	}
}

func randomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64()*2 - 1 // Random values between -1 and 1
		}
	}
	return matrix
}
