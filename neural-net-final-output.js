// Neural Net Output

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
        loss += target * Math.log(prediction);
    }

    return -loss;
}

/* Takes in an array of predictions, and an array of targets
 Predictions and targets are also arrays
*/
function crossEntropyLossBatch(batchPredictions, batchTargets) {
    let totalLoss = 0;
    for (let i = 0; i < batchPredictions.length; i++) {
        const predictions = batchPredictions[i];
        const targets = batchTargets[i];
        totalLoss += crossEntropyLoss(predictions, targets);
    }
    return totalLoss / batchPredictions.length; // Average loss over the batch
}




/* returns Jacobian matrix based on backpropagation algorithm of batchPredictions, batchTargets
* This is backpropagation through crossentropy loss and softmax functions combined
*/
function finalOutputBackPropBatch(batchPredictions, batchTargets) {

}

module.exports = { 
    forwardPass: {
        crossEntropyLossBatch,
        softmaxBatch,
    },
    backProp: {
         finalOutputBackPropBatch,
    }
}
