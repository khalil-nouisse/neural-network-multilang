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

func sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}