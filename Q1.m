%%
%%Generate projected data
%X = standardize(train) %Standardize all data
X = train
[W, Z, evals, Xrecon, mu] = pcaPmtk(X, 1) %Extract one feature
%Z contains the projected data (centered)
%mu contains the centres
Y = train_labels

A_POS = find(Y == 1);
B_POS = find(Y == 2);

figure(1);clf;
plot3(X(A_POS,1), X(A_POS,2), X(A_POS, 3),'bx', 'markersize', 0.5);
hold on
plot3(X(B_POS,1), X(B_POS,2), X(B_POS, 3), 'ro', 'markersize', 0.5);
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')
% 
% Z2 = [min(Z); max(Z)]; % span the range
% Xrecon2 = Z2*W' + repmat(mu, 2,1); %extreme points projected onto space
% h=line([Xrecon2(1,1) Xrecon2(2,1)], [Xrecon2(1,2) Xrecon2(2,2)], [Xrecon2(1,3) Xrecon2(2,3)]);
% set(h,'linewidth', 1, 'color', 'k');


s  = 5;
wPCA = W
h=line([mu(1)-s*wPCA(1) mu(1)+s*wPCA(1)], [mu(2)-s*wPCA(2) mu(2)+s*wPCA(2)], [mu(3)-s*wPCA(3) mu(3)+s*wPCA(3)]);
set(h, 'color', 'k', 'linewidth', 1, 'linestyle', '--')
printPmtkFigure PCATrainingFigure

%%
%%Performing regression and classifying
%W is the space to project on
pcaProjection = train*W %Projected data
linear_model = linregFit(pcaProjection, train_labels)

%Performing on test data
pcaPrediction = linregPredict(linear_model, test*W)
pcaPrediction(pcaPrediction>1.5,1)=2
pcaPrediction(pcaPrediction<=1.5,1)=1

%Print performance
performance(test_labels, pcaPrediction)


%%
%%Plotting test dataset in 3D
X1 = test
Y1 = test_labels

A1_POS = find(Y1 == 1);
B1_POS = find(Y1 == 2);

figure(2);clf;
plot3(X(A1_POS,1), X(A1_POS,2), X(A1_POS, 3),'bx', 'markersize', 2);
hold on
plot3(X(B1_POS,1), X(B1_POS,2), X(B1_POS, 3), 'ro', 'markersize', 1);
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')

h1=line([mu(1)-s*wPCA(1) mu(1)+s*wPCA(1)], [mu(2)-s*wPCA(2) mu(2)+s*wPCA(2)], [mu(3)-s*wPCA(3) mu(3)+s*wPCA(3)])
set(h1, 'color', 'k', 'linewidth', 1, 'linestyle', '--')
%%TODO : ADD SEPARATION pcaBoundary
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')
printPmtkFigure PCATestingFigure

%%
%Plotting projected dataset
%Solution for pcaBoundary A*Z + B = 1.5
pcaBoundary = (1.5 - reg_output(1,1))/reg_output(2,1)

figure(3);clf;
plot(Q(A1_POS),test_labels(A1_POS),'bx','markersize',3)
hold on
plot(Q(B1_POS),test_labels(B1_POS),'ro','markersize',1.5)
hold on
refline(reg_output(2,1),reg_output(1,1)) 
hold on
yL = get(gca,'YLim');
h3 = line([pcaBoundary,pcaBoundary],[0.8,2.2])
set(h3, 'linewidth',2,'color','k','linestyle','--')

ylim([0.8,2.2])
xlabel('Extracted Feature'); 
printPmtkFigure PCAClassifiedFigure

%%
%%LDA
%%Setting means
X = train

mean_A = mean(train(A_POS,:))
mean_B = mean(train(B_POS,:))
centroid = (mean_A + mean_B)/2
fisher = fisherLdaFit(X, Y)
fisher_norm = fisher/norm(fisher)

figure(4);clf;
plot3(X(A_POS,1), X(A_POS,2), X(A_POS, 3),'bx', 'markersize', 2);
hold on
plot3(X(B_POS,1), X(B_POS,2), X(B_POS, 3), 'ro', 'markersize', 1);
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')
h4 = line([mu(1)-s*fisher_norm(1) mu(1)+s*fisher_norm(1)], [mu(2)-s*fisher_norm(2) mu(2)+s*fisher_norm(2)], [mu(3)-s*fisher_norm(3) mu(3)+s*fisher_norm(3)]);
set(h4, 'color', 'k', 'linewidth', 1, 'linestyle', ':','YLimInclude', 'off', 'XLimInclude', 'off')


%%TODO : ADD SEPARATION pcaBoundary
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')
printPmtkFigure LDATrainingFigure

%%
%%Plotting 3D Test data
figure(5);clf;
plot3(X(A1_POS,1), X(A1_POS,2), X(A1_POS, 3),'bx', 'markersize', 2);
hold on
plot3(X(B1_POS,1), X(B1_POS,2), X(B1_POS, 3), 'ro', 'markersize', 1);
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')
h4 = line([mu(1)-s*fisher(1) mu(1)+s*fisher(1)], [mu(2)-s*fisher(2) mu(2)+s*fisher(2)], [mu(3)-s*fisher(3) mu(3)+s*fisher(3)]);
set(h4, 'color', 'k', 'linewidth', 1, 'linestyle', ':','YLimInclude', 'off', 'XLimInclude', 'off')


%%TODO : ADD SEPARATION pcaBoundary
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')
printPmtkFigure LDATestingFigure

%%
%%Projecting onto LDA
ldaProjection = test*fisher_norm

%Boundary is midway between two means
ldaBoundary = centroid*fisher_norm
if mean_A > mean_B
    upperClass = 1
else
    upperClass = 2
end
lowerClass = 3 -upperClass
class_ldaPrediction(ldaProjection>=ldaBoundary,1) = upperClass
class_ldaPrediction(ldaProjection<ldaBoundary,1) = lowerClass

performance(test_labels, class_ldaPrediction)

%%
%Plotting 1D test data

figure(6);clf;
plot(ldaProjection(A1_POS),0,'bx','markersize',3)
hold on
plot(ldaProjection(B1_POS),0,'ro','markersize',1.5)
hold on
ylim([-0.2, 0.2])
h6 = line([ldaBoundary,ldaBoundary],[-0.2,0.2])
set(h6, 'linewidth',2,'color','k','linestyle','--')

xlabel('Extracted Feature'); 
printPmtkFigure LDAClassifiedFigure

