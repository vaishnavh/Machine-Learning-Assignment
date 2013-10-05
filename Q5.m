%%
%Logistic regression
logModel = logregFit(images, types, 'lambda', 0)
log_predicted = logregPredict(logModel,test_images)
performance(log_predicted, test_types, [1,2,3,4])




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
forest_index = (types == 2)
mountain_index = (types == 4)
train_index = forest_index | mountain_index
train_label = types(train_index)
train_label(train_label == 4) = 1
train_label(train_label == 2) = -1
log_train = norm_images(train_index,:)


%%
%%Saving test data
forest_index = (test_types == 2)
mountain_index = (test_types == 4)
test_index = forest_index | mountain_index
test_label = test_types(test_index)
test_label(test_label == 4) = 1
test_label(test_label == 2) = -1
log_test = norm_test_images(test_index,:)


%%
%%Splitting
logCV = cvpartition(train_label, 'k', 5)

%%
%%Generating sample model
log_lambdas  = [0.001, 0.01, 0.1, 1, 10]
log_error = zeros(length(log_lambdas),1)
for j = 1:length(log_lambdas)
    log_lambda = log_lambdas(j)
    for i = 1:logCV.NumTestSets
        trIdx = logCV.training(i)
        teIdx = logCV.test(i)
        mmwrite('norm_images',log_train(trIdx,:))
        mmwrite('types',train_label(trIdx))
        mmwrite('norm_test_images', log_train(teIdx,:))
        mmwrite('test_types',train_label(teIdx))
        command = ['./l1_logreg_train -s norm_images types ' num2str(log_lambda) ' l1_logreg_model']
        system(command)
        system('./l1_logreg_classify -t test_types l1_logreg_model norm_test_images  l1_logreg_result')
        l1_logreg_result = mmread('l1_logreg_result')
        log_error(j) = log_error(j) + sum(~(l1_logreg_result == train_label(teIdx))) 
    end
    log_error(j) = log_error(j)/logCV.NumTestSets    
    
end

