%%
%%SVM Linear Kernel
%%Doesn't require validation
model1 = libsvmtrain(types, norm_images, '-t 0')

[model1_label, accuracy, prob_estimates] = libsvmpredict(test_types, norm_test_images, model1)

%%

%%Generating splits
%CV = cvpartition(types, 'k', 5)
%%
%%SVM Polynomial Kernel
degrees = [1,2,3,4,5,7,9];
coeffs = [0, 1, 10, 100, 1000];
errors = zeros(length(degrees), length(coeffs));
for i = 1:length(degrees);
    for j = 1:length(coeffs);
        errors(i,j) = polynomial_kernel([coeffs(j), degrees(i)], CV, norm_images, types);        
    end
end

%%
poly_min = fminsearch(@(x) polynomial_kernel(x,CV,norm_images,types),[1,2])

%%
condition = ['-t 1 -d ' num2str(2.0750) ' -g 1 -r ' num2str(0.9750)]
poly_model = libsvmtrain(types, norm_images, condition)
poly_prediction = libsvmpredict(test_types, norm_test_images, poly_model)
poly_error = sum(~(poly_prediction == test_types)) 
