package network

// Train runs one step of feedforward and then backpropagation to update the weights.
func (nn *NeuralNetwork) Train(inputs []float64, targets []float64) {
	// 1. Forward Pass
	activatedHidden, activatedOutput := nn.Feedforward(inputs)

	// 2. Backward Pass
	nn.backwardpass(inputs, targets, activatedOutput, activatedHidden)
}
