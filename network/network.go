package network

import (
	"math/rand"
	"time"
)

type NeuralNetwork struct {
	InputNodes  int
	OutputNodes int
	HiddenNodes int

	WeightsInputHidden  [][]float64
	WeightsHiddenOutput [][]float64

	BiasHidden [][]float64
	BiasOutput [][]float64

	LearningRate float64
}

func NewNetwork(inputs, hidden, outputs int, learningRate float64) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	return &NeuralNetwork{
		InputNodes:  inputs,
		OutputNodes: outputs,
		HiddenNodes: hidden,

		WeightsInputHidden:  randomMatrix(inputs, hidden),
		WeightsHiddenOutput: randomMatrix(hidden, outputs),

		BiasHidden: randomMatrix(hidden, 1),
		BiasOutput: randomMatrix(outputs, 1),

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
