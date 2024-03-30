clear
weight = readmatrix('data/Parameters/checkpoints.csv');
trainInput = readmatrix('data/TestingSet/soo.csv');

%% Synthetic Testing Data
% x = (10:1:100) +10;
% y = cos(10:1:100).*5;
% g = ((10:1:100)./10).*5;
% for i = 1:length(y)
%     y(i) = y(i) - rand(1) * 10;
% end

x = trainInput(:,1)';
y = trainInput(:,2)';
g = trainInput(:,3)';
test_data = [x',y',g'];
[r,c] = size(test_data);

%% Neural Network
Weights = {
        weight(:,1)';
        weight(:,2)';
        };
Bias = {
    weight(:,3)';
    weight(:,4)';
};
networkOut = [];
for index = 1:r
    %Layer 1
    for i = 1:length(Weights{1})
        neuron = Weights{1}(i) * sum(test_data(index,1:c-1))  + Bias{1}(i);
        Network{1}(i) = neuron;
    end
    
    %Layer 2
    for i = 1:length(Weights{2})
        neuron = Weights{2}(i) * sum(Network{1}) + Bias{2}(i);
        Network{2}(i) = neuron;
    end
    networkOut(index) = sum(Network{2});
end
%%
figure
plot(x, test_data(:,c), "-k")
hold on
plot(x, networkOut, ".r")
grid on
title("Neural prediction with Input 1 compared to Ground Truth")
xlabel("Input1")
ylabel("Output")
legend("True", "Neural Network")

figure
plot(y, test_data(:,c), "-k")
hold on
plot(y, networkOut, ".b")
grid on
title("Neural prediction with Input 2 compared to Ground Truth")
xlabel("Input2")
ylabel("Output")
legend("True", "Neural Network")

clear