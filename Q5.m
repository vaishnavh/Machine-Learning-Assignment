%%
%Logistic regression
logModel = logregFit(images, types, 'lambda', 0)
log_predicted = logregPredict(logModel,test_images)
performance(log_predicted, test_types, [1,2,3,4])


%%

%%Generating splits
CV = cvpartition(types, 'k', 10)

%%
%%
%%Fitting model
lambdas = 1:10
logreg_errors = zeros(size(lambdas))
for lambda = lambdas    
    for i = 1:CV.NumTestSets
        trIdx = CV.training(i)
        teIdx = CV.test(i)
        logModel = logregFit(images(trIdx,:), types(trIdx,:), 'lambda', lambda)
        log_predicted = logregPredict(logModel,images(teIdx,:))
        [er] = sum(teIdx) - performance(log_predicted, types(teIdx,:), [1,2,3,4])
        logreg_errors(1, lambda) = logreg_errors(1, lambda) + er
    end
end
    


%%
%%Logistic regression L1 : Saving train data
mmwrite('norm_images',norm_images)
mmwrite('types',types)

%%
%%Saving test data
mmwrite('norm_test_images', norm_test_images)
mmwrite('test_types', test_types)

%%
%%Generating sample model

system('./l1_logreg_train -s norm_images types 0.01 l1_logreg_model')
system('./l1_logreg_classify -t test_types l1_logreg_model norm_test_images  l1_logreg_result')
l1_logreg_result = mmread('l1_logreg_result')
