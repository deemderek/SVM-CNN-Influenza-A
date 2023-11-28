function [predictionModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'LocationInSequence', 'Strain1Base', 'Strain2Base', 'Strain3Base'};
predictors = inputTable(:, predictorNames);
response = inputTable.Strain4Base;
isCategoricalPredictor = [false, false, false, false];

regressionNeuralNetwork = fitrnet(...
    predictors, ...
    response, ...
    'LayerSizes', 100, ...
    'Activations', 'relu', ...
    'Lambda', 0, ...
    'IterationLimit', 1000, ...
    'Standardize', true);

predictorExtractionFcn = @(t) t(:, predictorNames);
neuralNetworkPredictFcn = @(x) predict(regressionNeuralNetwork, x);
predictionModel.predictFcn = @(x) neuralNetworkPredictFcn(predictorExtractionFcn(x));

predictionModel.RequiredVariables = {'LocationInSequence', 'Strain1Base', 'Strain2Base', 'Strain3Base'};
predictionModel.RegressionNeuralNetwork = regressionNeuralNetwork;
predictionModel.About = 'This struct is a trained model exported from Regression Learner R2021a.';

inputTable = trainingData;
predictorNames = {'LocationInSequence', 'Strain1Base', 'Strain2Base', 'Strain3Base'};
predictors = inputTable(:, predictorNames);
response = inputTable.Strain4Base;
isCategoricalPredictor = [false, false, false, false];

validationPredictions = predict(predictionModel.RegressionNeuralNetwork, predictors);
validationRMSE = sqrt(resubLoss(predictionModel.RegressionNeuralNetwork, 'LossFun', 'mse'));
end
