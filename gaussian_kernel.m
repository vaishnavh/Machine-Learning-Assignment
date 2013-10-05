function [ average_error ] = gaussian_kernel( parameters, CV, input, output)
%UNTITLED function [ average_error ] = gaussian_kernel( parameters, CV, input, output)
    cost = parameters(1)
    gamma = parameters(2)
    error = 0
    condition = ['-t 2 -g ' num2str(gamma) ' -c ' num2str(cost)]

    for i = 1:CV.NumTestSets
        trIdx = CV.training(i)
        teIdx = CV.test(i)
        temp_model = libsvmtrain(output(trIdx,:), input(trIdx,:), condition)
        prediction = libsvmpredict(output(teIdx,:), input(teIdx,:), temp_model)
        error = error + sum(~(prediction == output(teIdx,:)))             
    end
    average_error = (error)/CV.NumTestSets
end
