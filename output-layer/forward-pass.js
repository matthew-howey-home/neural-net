// Functions for Forward Pass through final layer 

// "Sample" refers to an array, with one element for each class the neural net is trying to predict
// Only one class is correct, a targetSample has value of 1.0 for the correct class, 0.0 for all others ('one hot encoding')
// "Batch" refers to an array of samples

// "Logits" is the output from the final layer before the output layer
// "Predictions" is the Logits transformed via softmax into a form which attempts to predict the correct class
// "Target" is the correct class as a one hot coded array

/* Takes in Logits Sample which is an array
With one element for each class
Applies softmax function to sample
*/
function softmaxForSample(logitsSample) {
    const expValues = logitsSample.map(logitClass => Math.exp(logitClass));
    const sumExpValues = expValues.reduce((sum, expVal) => sum + expVal, 0);
    return expValues.map(expVal => expVal / sumExpValues);
}

/**
 * Takes in Array (Batch) of logit Samples
 * Each logit sample is an array with one element for each class
 * Returns an array of Logit Samples transformed by the softmax function
 */
function softmaxForBatch(batchOfLogitSamples) {
    return batchOfLogitSamples.map((logitsSample) => softmaxForSample(logitsSample));
}

/** 
 * Takes in a Prediction and Target Sample
 * Both Prediction and Target Samples are an array, with one element for each class
 * The Target is a one hot coded array, and the Prediction is an attempt to predict target as closely as possible
 * Returns a single loss value based on how close the Prediction sample is to Target sample overall
*/
function crossEntropyLossForSample(predictionSample, targetSample) {
    let loss = 0;

    // Loop over each class and compute the loss
    for (let i = 0; i < predictionSample.length; i++) {
        const targetClass = targetSample[i];
        const predictionClass = predictionSample[i];
        loss += targetClass * Math.log(predictionClass);
    }

    return -loss;
}

/* Takes in an Array (Batch) of prediction samples, and an Array (Batch) of target samples
   Both Prediction and Target Samples are an array, with one element for each class
   Returns a single loss value based on how close the Prediction samples are to Target samples overall
*/
function crossEntropyLossForBatch(batchOfPredictionSamples, batchOfTargetSamples) {
    let totalLoss = 0;
    for (let i = 0; i < batchOfPredictionSamples.length; i++) {
        const predictionSample = batchOfPredictionSamples[i];
        const targetSample = batchOfTargetSamples[i];
        totalLoss += crossEntropyLossForSample(predictionSample, targetSample);
    }
    return totalLoss / batchOfPredictionSamples.length; // Average loss over the batch
}

module.exports = { softmaxForBatch, crossEntropyLossForBatch };