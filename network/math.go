package network

import "math"

// dotProduct computes the matrix-vector product: result[j] = sum_i(inputs[i] * weights[i][j])
// weights is [inputNodes][outputNodes], inputs is [inputNodes], result is [outputNodes]
func dotProduct(inputs []float64, weights [][]float64) []float64 {
	inputLen := len(inputs)
	weightRows := len(weights)
	weightCols := len(weights[0])

	if inputLen != weightRows {
		panic("Matrix shape mismatch: inputs length must equal weights rows")
	}

	result := make([]float64, weightCols)

	for j := 0; j < weightCols; j++ {
		sum := 0.0
		for i := 0; i < weightRows; i++ {
			sum += inputs[i] * weights[i][j]
		}
		result[j] = sum
	}

	return result
}

// Activation Functions

// 1- Sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// 2 - ReLU
func relu(x float64) float64 {
	return math.Max(0, x)
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

//Loss Functions

func lossFunc(realValue float64, prediction float64) float64 {
	return math.Pow(realValue-prediction, 2)
}

func LossFuncDerivative(realValue float64, prediction float64) float64 {
	return 2 * (prediction - realValue)
}
