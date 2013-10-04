function [ input_weights, hidden_weights ] = neural_newtork( input_size, hidden_size, output_size, epochs, train_data, train_labels)
%Returns a Neural Network model
%   input_size, hidden_size, output_size, epochs, train_data, train_labels
IH_weight = (randn(input_size, hidden_size) - 0.5)/10
HO_weight = (randn(hidden_size, output_size) - 0.5)/10

for i = 1:epochs
    H_input = tansig(train_data*IH_weight)'
    p
end

end

