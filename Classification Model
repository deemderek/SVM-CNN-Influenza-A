function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)

inputTable = trainingData;
predictorNames = {'LocationInSequence', 'NitrogenousBase'};
predictors = inputTable(:, predictorNames);
response = ixnputTable.Strain;
isCategoricalPredictor = [false, true];
template = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 0.1692230898116188, ...
    'BoxConstraint', 757.0163629918361, ...
    'Standardize', false);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical({'CY046933'; 'FJ581447'; 'GU056986'; 'GU057032'}));
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.RequiredVariables = {'LocationInSequence', 'NitrogenousBase'};
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2021a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T)')
inputTable = trainingData;
predictorNames = {'LocationInSequence', 'NitrogenousBase'};
predictors = inputTable(:, predictorNames);
response = inputTable.Strain;
isCategoricalPredictor = [false, true];

[validationPredictions, validationScores] = predict(trainedClassifier.ClassificationSVM, predictors);
validationAccuracy = 1 - resubLoss(trainedClassifier.ClassificationSVM, 'LossFun', 'ClassifError');
end
