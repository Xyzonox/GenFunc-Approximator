
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural network that can approximate both single variable and
% multivariable continuous functions. Neural Network has an input layer,
% two hidden layers and an output layer, with an arbitrary amount of nodes
% (dictated by the amount of weights in the checkpoints file). Layer 1
% nodes use the function (X^W1 + B1, X is the sum of inputs, W1 is the weight 
% of a Layer 1 node, and B1 is a bias of a Layer 1 node) and Layer 2 nodes
% use the function (X*W2 + B1, X is the sum of Layer 1 nodes, W2 is the 
% weight of a Layer 2 node, and B2 is a bias of a Layer 2 node).
%
%
% Author: Zion Hackett
% Contact: zionmjh@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data and Weights
clear
weight = readmatrix('data/Parameters/checkpoints.csv');
trainInput = readmatrix('data/TrainingSet/soo.csv');

train_data = [];
%% Synthetic Training Data
% x = (10:1:100) +10;
% y = cos(10:1:100).*5;
% g = ((10:1:100)./10).*5;
% for i = 1:length(y)
%     y(i) = y(i) - rand(1) * 10;
% end

 x = trainInput(:,1)';
 y = trainInput(:,2)';
 g = trainInput(:,3)';

train_data = [x',y',g'];
[r,c] = size(train_data);

Weights = {
        weight(:,1)';
        weight(:,2)';
        };
Bias = {
    weight(:,3)';
    weight(:,4)';
};
for i = 1:2
    for j = 1:length(Weights{i})
        Weights{i}(j) = (rand(1))* .01;
        Bias{i}(j) = (rand(1))* .01;
        if round(rand(1)) 
            %Weights{i}(j) = Weights{i}(j) *-1;
            %Bias{i}(j) = Bias{i}(j) * -1;
        end
    end
end

%% Neural Network Superparameters
lossStore(1) = 6000; %Starting loss to initiate loop
meanError = 1;
trainStep = .00000005; %Training Steps

count = 0; %Time Network within acceptable loss
stagnant = 0; %Time Network not changing

lossAllowance = .5; %Acceptable amount of loss
errorTime = 2; %Acceptable amount of time within Acceptable loss
epoch = 1; %Iterations of training loop
index = 10; %Index of the dataset
range = 2; %Hubert Loss parameter
leak = .4; %Leaky relu multiplyer

WeightError = {}; %Stores change in loss attributed to Weights
Network = {}; %Stores value of each neuron

%% Neural Network Training
while count < errorTime
    reference = train_data(index,c);
    %%Hidden Layers
    %Layer 1
    input = sum((train_data(index,1:c-1)));
    for i = 1:length(Weights{1})
        neuron = input * Weights{1}(i)  + Bias{1}(i);
        if neuron < 0
            neuron = input * leak * Weights{1}(i)  + Bias{1}(i);
        end
        Network{1}(i) = neuron;
    end
    layerOne(index) = sum(Network{1});
    %Layer 2
    [r,c] = size(train_data);
    for i = 1:length(Weights{2})
        neuron = Weights{2}(i) * sum(Network{1}) + Bias{2}(i);
        if neuron < 0
            neuron = Weights{2}(i) * leak * sum(Network{1}) + Bias{2}(i);
        end
        Network{2}(i) = neuron;
    end
    layerTwo(index) = sum(Network{2});

    %Output Layer
    networkOut = sum(Network{2});
    networkOutVec(index) = networkOut;

    error(index) = (reference - networkOut);
    squareError(index) = error(index)^2 * 2;

    %% Loss Weight Update
    index = index + 1;
    if index > r
        randIndex = randi(length(error));
        %randIndex = find(max(error));
        randInput = sum((train_data(randIndex,1:c-1)));

        %%Backpropogation  
        for i = 1:r
            if abs(error(i)) <= range || true
                lossV(i) = (squareError(i)) ;
                backprop_loss(i) = (-1 * error(i));
            else
                lossV(i) = (range * (error(i) - 0.5 * range)) ;
                backprop_loss(i) = -1 * range * networkOutVec(i);
            end
        end
        loss = sum(lossV) * 1/r;

        %Change in loss 
        if epoch > 1
            deviLoss = (lossStore(epoch-1) - loss);
        else
            deviLoss = 1;
        end
        lossStore(epoch) = loss;
        meanError = mean(error);
            %Loss Calculation
        for i = 1:length(Weights{2})
            WeightError{2}(i) = sum(backprop_loss .* layerOne)*1/r;
            BiasError{2}(i) = sum(backprop_loss)*1/r;
            if Network{2}(i) < 0 
                WeightError{2}(i) = sum(backprop_loss .* layerOne * leak)*1/r;
            end
        end
        for i = 1:length(Weights{1})
            intermediate = [];
            for j = 1:r
                intermediate(j) = backprop_loss(j) .* sum(Weights{2}.* sum(train_data(j,1:c-1)));
            end
            %WeightError{1}(i) = sum(backprop_loss.* Weights{2}.* randInput);
            WeightError{1}(i) = sum(intermediate) * 1/r;
            BiasError{1}(i) = sum(backprop_loss.* sum(Weights{2})) * 1/r;
            if Network{1}(i) < 0 
                WeightError{1}(i) = WeightError{1}(i) * leak;
            end
        end   
        %Loss Application
        for i = 1:length(Weights{1})
            Weights{1}(i) = Weights{1}(i) - trainStep * WeightError{1}(i);
            Bias{1}(i) = Bias{1}(i) - trainStep * BiasError{1}(i);
        end
        for i = 1:length(Weights{2})
            Weights{2}(i) = Weights{2}(i) - trainStep * WeightError{2}(i);
            Bias{2}(i) = Bias{2}(i) - trainStep * BiasError{2}(i);
        end
        %

        clc
        fprintf("\n")
        fprintf("Current Loss: %.4f \n", loss)
        fprintf("Current Deviance in Loss: %.6f \n", deviLoss)
        fprintf("Max Error: %.4f \n", max(abs(error)))
        fprintf("Accepted Iterations: %.2f \n", count)
        fprintf("Epoch: %.0f \n", epoch)

        if deviLoss < 0 
            for i = 1:2
                for j = 1:length(Weights{i})
                    %Weights{i}(j) = Weights{1}(i) + trainStep * WeightError{1}(i) * 1 * rand(1);
                    %Bias{i}(j) = Bias{1}(i) + trainStep * BiasError{1}(i) * 1 * rand(1);
                    if round(rand(1))
                        %Weights{i}(j) = Weights{i}(j) *-1;
                        %Bias{i}(j) = Bias{i}(j) * -1;
                    end
                end
            end
        end

        %Time out of Loss
        if loss < lossAllowance || deviLoss < .0000005
            count = count + 1;
        else
            count = 0;
        end

        epoch = epoch + 1;
        index = 1;
    end
    pause(.00001)
end
%% Results
writematrix([Weights{1}' Weights{2}' Bias{1}' Bias{2}'], "data/Parameters/checkpoints.csv")

figure
plot(x, train_data(:,c), "-k")
hold on 
plot(x, networkOutVec, ".r")
title("Neural prediction with Input 1 compared to Ground Truth")
xlabel("Input1")
ylabel("Output")
legend("True", "Neural Network")

figure
plot(y, train_data(:,c), "-k")
hold on 
plot(y, networkOutVec, ".b")
title("Neural prediction with Input 2 compared to Ground Truth")
xlabel("Input2")
ylabel("Output")
legend("True", "Neural Network")

figure
plot(1:length(lossStore), (lossStore), "-k")
title("Learning Curve")
xlabel("Epoch #")
ylabel("Log Loss")


