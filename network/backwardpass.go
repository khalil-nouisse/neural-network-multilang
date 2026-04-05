package network

func (nn *NeuralNetwork) backwardpass(targets []float64, activations [][]float64) {
	numLayers := len(nn.LayerSizes)

	// deltas[l] will store the error term for layer l+1
	deltas := make([][]float64, numLayers-1)

	// Calculate delta for the output layer
	outputActivations := activations[numLayers-1]
	outputLayerIdx := numLayers - 2

	deltas[outputLayerIdx] = make([]float64, nn.LayerSizes[numLayers-1])
	for i := 0; i < nn.LayerSizes[numLayers-1]; i++ {
		deltas[outputLayerIdx][i] = LossFuncDerivative(targets[i], outputActivations[i]) * sigmoidDerivative(outputActivations[i])
	}

	// Calculate deltas for hidden layers (going backwards)
	for l := numLayers - 3; l >= 0; l-- {
		deltas[l] = make([]float64, nn.LayerSizes[l+1])
		nextDeltas := deltas[l+1]

		for i := 0; i < nn.LayerSizes[l+1]; i++ {
			errorSum := 0.0
			for j := 0; j < nn.LayerSizes[l+2]; j++ {
				errorSum += nn.Weights[l+1][i][j] * nextDeltas[j]
			}
			deltas[l][i] = errorSum * sigmoidDerivative(activations[l+1][i])
		}
	}

	// Gradient update for all layers
	for l := 0; l < len(nn.Weights); l++ {
		for i := 0; i < nn.LayerSizes[l]; i++ {
			for j := 0; j < nn.LayerSizes[l+1]; j++ {
				gradient := deltas[l][j] * activations[l][i]
				nn.Weights[l][i][j] -= nn.LearningRate * gradient
			}
		}
		for j := 0; j < nn.LayerSizes[l+1]; j++ {
			nn.Biases[l][j][0] -= nn.LearningRate * deltas[l][j]
		}
	}
}
