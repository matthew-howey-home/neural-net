const { forwardPass, backPropagation } = require('.');

// Example usage
const batchOfLogits = [
    [2.0, 1.0, 0.1, 0.5],  // Logits for flower 1
    [0.5, 1.5, 1.0, 0.7],  // Logits for flower 2
    [9.5, 2.3, 0.1, 0.9],  // Logits for flower 3
    // Add more flowers to the batch...
];

const batchOfTargets = [
    [0, 1, 0, 0],  // One-hot encoded true label for flower 1
    [0, 0, 1, 0],  // One-hot encoded true label for flower 2
    [1, 0, 0, 0],  // One-hot encoded true label for flower 3
    // Add more labels for the rest of the batch...
];

const batchOfPredictions = forwardPass.softmaxForBatch(batchOfLogits);
const loss = forwardPass.crossEntropyLossForBatch(batchOfPredictions, batchOfTargets);

console.log('Batch of Logits:', batchOfLogits);
console.log("Batch of Predictions:", batchOfPredictions);
console.log('Batch of Targets', batchOfTargets);
console.log("Loss:", loss);

const gradients = backPropagation.finalLayerBackPropagationForBatch(batchOfPredictions, batchOfTargets);
const gradientsByLearningStep = backPropagation.gradientsByLearningRate(gradients, 0.1);
const transformedLogits = backPropagation.logitsTransformedByLearningRate(batchOfLogits, gradientsByLearningStep);

console.log("gradients:", gradients);
console.log("Gradients Transformed by Learning Rate:", gradientsByLearningStep);
console.log("Logits Transformed by Learning Rate:", transformedLogits);

const revisedBatchPredictions = forwardPass.softmaxForBatch(transformedLogits);
const revisedLoss = forwardPass.crossEntropyLossForBatch(revisedBatchPredictions, batchOfTargets);

console.log("Revised Batch Predictions:",  revisedBatchPredictions);
console.log("Revised Loss:", revisedLoss);