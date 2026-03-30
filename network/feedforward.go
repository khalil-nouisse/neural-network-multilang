package network

// Feedforward runs the forward pass of the neural network and returns the output activations.
func (nn *NeuralNetwork) Feedforward(inputs []float64) []float64 {

	// Input → Hidden: dot product + bias + sigmoid
	rawHidden := dotProduct(inputs, nn.WeightsInputHidden)
	activatedHidden := make([]float64, len(rawHidden))

	for i := range rawHidden {
		z := rawHidden[i] + nn.BiasHidden[i][0]
		activatedHidden[i] = sigmoid(z)
	}

	// Hidden → Output: dot product + bias + sigmoid
	rawOutput := dotProduct(activatedHidden, nn.WeightsHiddenOutput)
	activatedOutput := make([]float64, len(rawOutput))

	for i := range rawOutput {
		a := rawOutput[i] + nn.BiasOutput[i][0]
		activatedOutput[i] = sigmoid(a)
	}

	return activatedOutput
}