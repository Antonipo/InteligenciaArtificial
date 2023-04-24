function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function stepFunction(z) {
  return z >= 0 ? 1 : 0;
}

function tanh(z) {
  return (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
}

function relu(z) {
  return Math.max(0, z);
}

function perceptronActivationFunction(
  inputs,
  weights,
  bias,
  activationFunction
) {
  let weightedSum = 0;

  for (let i = 0; i < inputs.length; i++) {
    weightedSum += inputs[i] * weights[i];
  }

  weightedSum += bias;

  switch (activationFunction) {
    case "sigmoid":
      return sigmoid(weightedSum);
    case "step":
      return stepFunction(weightedSum);
    case "tanh":
      return tanh(weightedSum);
    case "relu":
      return relu(weightedSum);
    default:
      throw new Error("Invalid activation function specified.");
  }
}