function softmax(logits) {
    const expValues = logits.map(logit => Math.exp(logit));
    const sumExpValues = expValues.reduce((sum, expVal) => sum + expVal, 0);
    return expValues.map(expVal => expVal / sumExpValues);
}

/**
 * Takes in array of logits
 * Logits is itself an array
 * Returns an array of logits transformed by the softmax function
 */
function softmaxBatch(batchLogits) {
    return batchLogits.map((logits) => softmax(logits));
}

/** 
 * Takes in a prediction and target in the form of one hot coded array
 * Returns a single loss value
*/
function crossEntropyLoss(predictions, targets) {
    let loss = 0;

    // Loop over each class and compute the loss
    for (let i = 0; i < predictions.length; i++) {
        const target = targets[i];
        const prediction = predictions[i];
        loss += target * Math.log(prediction + 1e-15); // Avoid log(0) by adding 1e-15
    }

    return -loss;
}

function crossEntropyLossBatch(batchPredictions, batchTargets) {
    let totalLoss = 0;
    for (let i = 0; i < batchPredictions.length; i++) {
        const predictions = batchPredictions[i];
        const targets = batchTargets[i];
        totalLoss += crossEntropyLoss(predictions, targets);
    }
    return totalLoss / batchPredictions.length; // Average loss over the batch
}

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

const batchPredictions = softmaxBatch(batchLogits);
const loss = crossEntropyLossBatch(batchPredictions, batchTargets);

console.log("Batch Predictions:", batchPredictions);
console.log("Loss:", loss);