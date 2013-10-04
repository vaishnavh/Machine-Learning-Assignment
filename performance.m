function [errors, accuracy, precision, recall, f_measure] = performance(test_labels, class_prediction, types)
%Prints the performance measure of a given test and predicted labels
dimension = size(types)
precision = zeros(dimension)
recall = zeros(dimension)
f_measure = zeros(dimension)
for type = types
    classified = (class_prediction == type)
    belong_to = (test_labels == type)
    precision(1, type) = sum(classified & belong_to)/sum(classified)
    recall(1, type) = sum(classified & belong_to)/sum(belong_to)
    f_measure(1, type) = harmmean([precision(1,type), recall(1, type)])
end
accuracy = sum(test_labels == class_prediction)
end

