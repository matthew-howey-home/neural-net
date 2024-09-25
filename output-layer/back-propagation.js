
/* backpropagation for one sample of predictions and targets
* This is backpropagation though Cross Entropy and Softmax combined
*/
function finalLayerBackPropagation(predictionSample, targetSample) {
    const sampleGradient = [];
    for (let i = 0; i < predictionSample.length; i++) {
        sampleGradient.push(predictionSample[i] - targetSample[i]);
    }
    return sampleGradient
}


/* returns array based on backpropagation algorithm of batchPredictions, batchTargets
* This is backpropagation through crossentropy loss and softmax functions combined
*/
function finalLayerBackPropagationForBatch(batchOfPredictionSamples, batchOfTargetSamples) {
    const numberOfSamples = batchOfPredictionSamples.length;
    const numberOfClasses = batchOfPredictionSamples[0].length;

    const batchOfGradients = [];
    for (let sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++) {
        const sampleGradient = finalLayerBackPropagation(batchOfPredictionSamples[sampleIndex], batchOfTargetSamples[sampleIndex]);
        batchOfGradients.push(sampleGradient);
    }

    // final gradient is an average of the gradients in each class
    const finalGradients = [];
    for (let classIndex = 0; classIndex < numberOfClasses; classIndex++) {
        let sumOfClassValues = 0;
        for (let sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++) {
            const gradient = batchOfGradients[sampleIndex];
            sumOfClassValues += gradient[classIndex];
        }
        finalGradients.push(sumOfClassValues / numberOfSamples);
    }

    return finalGradients
}

// takes in array of gradients and multiplies each element by the learningRate
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
    finalLayerBackPropagationForBatch, gradientsByLearningRate, logitsTransformedByLearningRate,
}