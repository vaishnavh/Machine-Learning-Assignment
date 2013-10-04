%%
%%Getting data and initializing stuff
load('fisherIrisData')
X = meas(:,3:4) %Get only petal length and width
Y = canonizeLabels(species) %Mapping to integers

%%
%%LDA
lda_model = discrimAnalysisFit(X, Y, 'linear')
lda_h = plotDecisionBoundary(X, Y, @(Xtest)discrimAnalysisPredict(lda_model, Xtest));
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')

%%
%%QDA
qda_model = discrimAnalysisFit(X, Y, 'quadratic')
qda_h = plotDecisionBoundary(X, Y, @(Xtest)discrimAnalysisPredict(qda_model, Xtest));
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')

%%
%%RDA
%TODO  : Choose best alpha
rda_model = discrimAnalysisFit(X, Y, 'rda','lambda',0, 'gamma', 0)
rda_h = plotDecisionBoundary(X, Y, @(Xtest)discrimAnalysisPredict(rda_model, Xtest));
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')
