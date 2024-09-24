const { forwardPass, backProp } = require('./neural-net-final-output');

// Example usage
const batchLogits = [
    [2.0, 1.0, 0.1, 0.5],  // Logits for flower 1
    [0.5, 1.5, 1.0, 0.7],  // Logits for flower 2
    // Add more flowers to the batch...
];

const batchTargets = [
    [1, 0, 0, 0],  // One-hot encoded true label for flower 1
    [0, 1, 0, 0],  // One-hot encoded true label for flower 2
    // Add more labels for the rest of the batch...
];

const batchPredictions = forwardPass.softmaxBatch(batchLogits);
const loss = forwardPass.crossEntropyLossBatch(batchPredictions, batchTargets);

console.log('Batch Logits:', batchLogits);
console.log("Batch Predictions:", batchPredictions);
console.log('Batch Targets', batchTargets);
console.log("Loss:", loss);

const gradients = backProp.finalOutputBackPropBatch(batchPredictions, batchTargets);
const gradientsByLearningStep = backProp.gradientsByLearningRate(gradients, 0.1);
const transformedLogits = backProp.logitsTransformedByLearningRate(batchLogits, gradientsByLearningStep);

console.log("gradients:", gradients);
console.log("Gradients Transformed by Learning Rate:", gradientsByLearningStep);
console.log("Logits Transformed by Learning Rate:", transformedLogits);

const revisedBatchPredictions = forwardPass.softmaxBatch(transformedLogits);
const revisedLoss = forwardPass.crossEntropyLossBatch(revisedBatchPredictions, batchTargets);

console.log("Revised Batch Predictions:",  revisedBatchPredictions);
console.log("Revised Loss:", revisedLoss);