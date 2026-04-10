package network

// Feedforward runs the forward pass of the neural network and returns the activations
// of ALL layers (including input) and the final prediction.
func (nn *NeuralNetwork) Feedforward(inputs []float64) ([][]float64, []float64) {
	activations := make([][]float64, len(nn.LayerSizes))
	activations[0] = inputs

	currentActivation := inputs

	for l := 0; l < len(nn.Weights); l++ {
		raw := dotProduct(currentActivation, nn.Weights[l])
		nextActivation := make([]float64, len(raw))

		for j := range raw {
			z := raw[j] + nn.Biases[l][j][0]
			nextActivation[j] = sigmoid(z)
		}

		activations[l+1] = nextActivation
		currentActivation = nextActivation
	}

	return activations, currentActivation
}
