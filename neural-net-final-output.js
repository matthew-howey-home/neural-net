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

/* backpropagation for one sample of predictions and targets
*/
function finalOutputBackProp(predictions, targets) {
    const response = [];
    for (let i = 0; i < predictions.length; i++) {
        response.push(predictions[i] - targets[i]);
    }
    return response
}


/* returns array based on backpropagation algorithm of batchPredictions, batchTargets
* This is backpropagation through crossentropy loss and softmax functions combined
*/
function finalOutputBackPropBatch(batchPredictions, batchTargets) {
    const numberOfSamples = batchPredictions.length;
    const numberOfClasses = batchPredictions[0].length;

    const batchGradients = [];
    for (let sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++) {
        const gradient = finalOutputBackProp(batchPredictions[sampleIndex], batchTargets[sampleIndex]);
        batchGradients.push(gradient);
    }

    // final gradient is an average of the gradients in each class
    const finalGradients = [];
    for (let classIndex = 0; classIndex < numberOfClasses; classIndex++) {
        let sumOfClassValues = 0;
        for (let sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++) {
            const gradient = batchGradients[sampleIndex];
            sumOfClassValues += gradient[classIndex];
        }
        finalGradients.push(sumOfClassValues / numberOfSamples);
    }

    return finalGradients
}

// takes in array of gradients and multiples each element by the learningRate
function gradientsByLearningRate(gradients, learningRate) {
    return gradients.map((gradient) => gradient * learningRate);
}

/* takes in array of gradients transformed by learning rate
and an array logits, and modifies the logits */
function logitsTransformedByLearningRate(logits, transformedGradients) {
    const transformedLogits = [];
    for (let logitIndex = 0; logitIndex < logits.length; logitIndex++) {
        const logit = logits[logitIndex];
        transformedLogits.push(
            logit.map((logitClass, classIndex) => logitClass - transformedGradients[classIndex])
        );
    }
    return transformedLogits;
}

module.exports = { 
    forwardPass: {
        crossEntropyLossBatch,
        softmaxBatch,
    },
    backProp: {
        finalOutputBackPropBatch,
        gradientsByLearningRate,
        logitsTransformedByLearningRate,
    }
}
