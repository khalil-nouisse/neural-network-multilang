package network

func (nn *NeuralNetwork) backwardpass(inputs []float64, targets []float64, finalOutputs []float64, hiddenOutputs []float64) {

	//OUTPUT ERROR (Calculate δ^L)
	//delta output
	deltasOutput := make([]float64, nn.OutputNodes)

	for i := 0; i < nn.OutputNodes; i++ {
		deltasOutput[i] = LossFuncDerivative(targets[i], finalOutputs[i]) * sigmoidDerivative(finalOutputs[i])
	}

	//delta hidden
	deltasHidden := make([]float64, nn.HiddenNodes)
	for i := 0; i < nn.HiddenNodes; i++ {
		errorSum := 0.0
		for j := 0; j < nn.OutputNodes; j++ {
			errorSum += nn.WeightsHiddenOutput[i][j] * deltasOutput[j]
		}
		deltasHidden[i] = errorSum * sigmoidDerivative(hiddenOutputs[i])
	}

	//gradiant update HIDDEN -> OUTPUT
	//Weight
	for i := 0; i < nn.HiddenNodes; i++ {
		for j := 0; j < nn.OutputNodes; j++ {
			gradiant := deltasOutput[j] * hiddenOutputs[i]
			nn.WeightsHiddenOutput[i][j] -= nn.LearningRate * gradiant
		}
	}
	//Bias
	for j := 0; j < nn.OutputNodes; j++ {
		nn.BiasOutput[j][0] -= nn.LearningRate * deltasOutput[j]
	}

	// UPDATE LAYER l (Input -> Hidden Weights)
	//Weight
	for i := 0; i < nn.InputNodes; i++ {
		for j := 0; j < nn.HiddenNodes; j++ {
			gradiant := deltasHidden[j] * inputs[i]
			nn.WeightsInputHidden[i][j] -= nn.LearningRate * gradiant
		}
	}
	//Bias
	for j := 0; j < nn.HiddenNodes; j++ {
		nn.BiasHidden[j][0] -= nn.LearningRate * deltasHidden[j]
	}

}
