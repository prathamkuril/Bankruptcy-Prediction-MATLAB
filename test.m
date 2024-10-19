% Load the best Logistic Regression model
load('best_LR_model.mat', 'best_LR_model');

% Load the best Random Forest model
load('best_RF_model.mat', 'best_RF_model');

% Load X_test and y_test data from CSV files
X_test = readmatrix('X_test.csv');
y_test = readmatrix('y_test.csv');

% Predict using the Logistic Regression model
y_pred_LR = predict(best_LR_model, X_test);

% Convert predictions to binary classes (0 or 1) for Logistic Regression
y_pred_class_LR = round(y_pred_LR);

% Predict using the Random Forest model
y_pred_RF = predict(best_RF_model, X_test);

% Convert predictions to binary classes (0 or 1) for Random Forest
y_pred_class_RF = round(str2double(y_pred_RF));

% Evaluate the performance of Logistic Regression
confusion_LR = confusionmat(y_test, y_pred_class_LR);
% Visualize the confusion matrix for Logistic Regression
figure;
confusionchart(confusion_LR, {'Not Bankrupt', 'Bankrupt'}, 'Title', 'Confusion Matrix for Logistic Regression');

% Evaluate the performance of Random Forest
confusion_RF = confusionmat(y_test, y_pred_class_RF);
% Visualize the confusion matrix for Random Forest
figure;
confusionchart(confusion_RF, {'Not Bankrupt', 'Bankrupt'}, 'Title', 'Confusion Matrix for Random Forest');

%{
% Display confusion matrix for Logistic Regression
disp('Confusion Matrix for Logistic Regression:');
disp(confusion_LR);

% Display confusion matrix for Random Forest
disp('Confusion Matrix for Random Forest:');
disp(confusion_RF);
%}

% Calculate metrics for Logistic Regression
accuracy_LR = sum(y_pred_class_LR == y_test) / numel(y_test);
precision_LR = sum(y_pred_class_LR & y_test) / sum(y_pred_class_LR);
recall_LR = sum(y_pred_class_LR & y_test) / sum(y_test);
F1_score_LR = 2 * (precision_LR * recall_LR) / (precision_LR + recall_LR);

% Calculate metrics for Random Forest
accuracy_RF = sum(y_pred_class_RF == y_test) / numel(y_test);
precision_RF = sum(y_pred_class_RF & y_test) / sum(y_pred_class_RF);
recall_RF = sum(y_pred_class_RF & y_test) / sum(y_test);
F1_score_RF = 2 * (precision_RF * recall_RF) / (precision_RF + recall_RF);

% Display metrics for Logistic Regression
disp('Metrics for Logistic Regression:');
disp(['Accuracy: ', num2str(accuracy_LR)]);
disp(['Precision: ', num2str(precision_LR)]);
disp(['Recall: ', num2str(recall_LR)]);
disp(['F1 Score: ', num2str(F1_score_LR)]);

% Display metrics for Random Forest
disp('Metrics for Random Forest:');
disp(['Accuracy: ', num2str(accuracy_RF)]);
disp(['Precision: ', num2str(precision_RF)]);
disp(['Recall: ', num2str(recall_RF)]);
disp(['F1 Score: ', num2str(F1_score_RF)]);



