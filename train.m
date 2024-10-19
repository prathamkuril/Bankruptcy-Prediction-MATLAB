data = readtable('C:\Users\Acer\City, University of London\mini projects Sem-1\ML mini project\data.csv');

% Check the size of the dataset
size(data)

% Display basic statistics for individual columns if available
summary(data)

% Check for missing values
missingValues = ismissing(data);
missingSummary = sum(missingValues);
disp('Summary of Missing Values:');
disp(missingSummary);

% Check for duplicates
duplicateRows = data(ismember(data, unique(data, 'rows'), 'rows') == 0, :);
numDuplicates = height(duplicateRows);
disp(['Number of Duplicate Rows: ', num2str(numDuplicates)]);

% Separating X and y
X = data{:, 2:end}; % Exclude 'Bankrupt' as it will be y
y = data{:, 'Bankrupt'}; % Target variable


% SMOTE code starts here

% Display class counts before oversampling
class_counts_before = tabulate(y);
disp('Class Counts Before SMOTE:');
disp(class_counts_before(:, [1, 2]));

% Visualize class counts before SMOTE
figure;
bar(class_counts_before(:, 1), class_counts_before(:, 2), 'b');
xlabel('Class');
ylabel('Count');
title('Class Counts Before SMOTE');
xticks([0 1]); % Assuming binary classification, change if needed

% Separating minority and majority classes
minority_X = X(y == 1, :);
majority_X = X(y == 0, :);

% Implement SMOTE using K-nearest neighbors
num_minority = sum(y == 1);
num_majority = sum(y == 0);

num_synthetic_samples = num_majority - num_minority;

% Assuming k=5 for nearest neighbors
knn_model = fitcknn(X(y == 1, :), y(y == 1), 'NumNeighbors', 5);

% Generate synthetic samples using K-nearest neighbors
synthetic_samples = zeros(num_synthetic_samples, size(X, 2));

for i = 1:num_synthetic_samples
    idx = randi(size(minority_X, 1)); % Randomly select a sample from the minority class
    sample = minority_X(idx, :);
    
    % Find k-nearest neighbors
    [neighbors, ~] = knnsearch(minority_X, sample, 'K', 5);
    
    % Randomly select one of the nearest neighbors
    nearest_neighbor_idx = neighbors(randi(5));
    nearest_neighbor = minority_X(nearest_neighbor_idx, :);
    
    % Generate synthetic sample by interpolation
    synthetic_samples(i, :) = sample + rand(1) * (nearest_neighbor - sample);
end

% Combine original minority class with synthetic samples
X_balanced = [X(y == 0, :); synthetic_samples; X(y == 1, :)];
y_balanced = [zeros(num_majority, 1); ones(num_synthetic_samples, 1); ones(num_minority, 1)]; % minority class label is 1

% Display class counts after oversampling
class_counts_after = tabulate(y_balanced);
disp('Class Counts After SMOTE:');
disp(class_counts_after(:, [1, 2]));

% Visualize class counts after SMOTE
figure;
bar(class_counts_after(:, 1), class_counts_after(:, 2), 'r');
xlabel('Class');
ylabel('Count');
title('Class Counts After SMOTE');
xticks([0 1]); 

% Compute correlations between features and the target variable
correlations = corr([X_balanced, y_balanced]);

% Exclude the last column as it represents correlations with 'Bankrupt'
feature_correlations = correlations(1:end-1, end);

% Sort correlations in ascending order to get negatively correlated features
[~, idx_neg] = sort(feature_correlations, 'ascend');
top_negatively_correlated_features = idx_neg(1:10);

% Sort correlations in descending order to get positively correlated features
[~, idx_pos] = sort(feature_correlations, 'descend');
top_positively_correlated_features = idx_pos(1:10);

% Get column names of features
feature_names = data.Properties.VariableNames(2:end); % Exclude the target variable 'Bankrupt'

% Define the top 10 positively and negatively correlated features
top_positively_correlated = feature_names(top_positively_correlated_features);
top_negatively_correlated = feature_names(top_negatively_correlated_features);

% Selecting all 20 positively and negatively correlated features
all_selected_features = [top_positively_correlated, top_negatively_correlated];
selected_features = [top_positively_correlated, top_negatively_correlated];

% Get the indices of these features in the original dataset
selected_feature_indices = ismember(data.Properties.VariableNames(2:end), all_selected_features);

% Creating new X and y using all selected features
new_X = X_balanced(:, selected_feature_indices);
new_y = y_balanced;

% Displaying the selected features
disp('Selected Features:');
disp(all_selected_features);

% Displaying the size of new X and y
disp('Size of new X:');
disp(size(new_X));
disp('Size of new y:');
disp(size(new_y));

% Combine new_X and new_y
dataForHeatmap = [new_X, new_y];

% Calculate the correlation matrix
correlationMatrix = corr(dataForHeatmap);

% Display the correlation heatmap
figure;
heatmap(correlationMatrix, 'ColorbarVisible', 'on', 'Colormap',winter);
title('Correlation Heatmap for dataForHeatmap');

% Get column names of new_X
column_names = selected_features; % Assuming 'selected_features' contains the column names

% Calculate statistics for new_X
min_values = min(new_X);
max_values = max(new_X);
std_dev = std(new_X);
mean_values = mean(new_X);

% Create a table with statistics
stats_table = table(column_names', min_values', max_values', std_dev', mean_values', ...
    'VariableNames', {'Column_Name', 'Min', 'Max', 'Standard_Deviation', 'Mean'});

% Display the table
disp('Statistics for new_X:');
disp(stats_table);


k = 5; % Number of folds for k-fold cross-validation

cv = cvpartition(size(new_X, 1), 'KFold', k); % Creating k-fold partitions

% Initialize variables to store the best model and its metrics for Logistic Regression
best_LR_model = struct();
best_LR_metrics = struct('accuracy', 0, 'precision', 0, 'recall', 0, 'F1_score', 0);

% Initialize variables to store the best model and its metrics for Random Forest
best_RF_model = struct();
best_RF_metrics = struct('accuracy', 0, 'precision', 0, 'recall', 0, 'F1_score', 0);

% Initialize arrays to store performance metrics across folds for Logistic Regression
accuracy_vals_LR = zeros(k, 1);
precision_vals_LR = zeros(k, 1);
recall_vals_LR = zeros(k, 1);
F1_score_vals_LR = zeros(k, 1);

% Initialize arrays to store performance metrics across folds for Random Forest
accuracy_vals_RF = zeros(k, 1);
precision_vals_RF = zeros(k, 1);
recall_vals_RF = zeros(k, 1);
F1_score_vals_RF = zeros(k, 1);


for fold = 1:k
    train_indices = cv.training(fold);
    test_indices = cv.test(fold);

    X_train = new_X(train_indices, :);
    y_train = new_y(train_indices, :);
    X_test = new_X(test_indices, :);
    y_test = new_y(test_indices, :);

    % Train Logistic Regression model
    [B, FitInfo] = lasso(X_train, y_train, 'Lambda', 0.01, 'Standardize', true, 'Alpha', 1, 'CV', 5);
    mdl_LR = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link','logit');  


    % Predict on the test set for Logistic Regression
    y_pred_LR = predict(mdl_LR, X_test);
    y_pred_class_LR = round(y_pred_LR);

    % Calculate metrics for Logistic Regression evaluation
    current_accuracy_LR = sum(y_pred_class_LR == y_test) / numel(y_test);
    current_precision_LR = sum(y_pred_class_LR & y_test) / sum(y_pred_class_LR);
    current_recall_LR = sum(y_pred_class_LR & y_test) / sum(y_test);
    current_F1_score_LR = 2 * (current_precision_LR * current_recall_LR) / (current_precision_LR + current_recall_LR);

    % Collect metrics for Logistic Regression evaluation
    accuracy_vals_LR(fold) = current_accuracy_LR;
    precision_vals_LR(fold) = current_precision_LR;
    recall_vals_LR(fold) = current_recall_LR;
    F1_score_vals_LR(fold) = current_F1_score_LR;

    % Update best Logistic Regression model if current performance is better
    if current_F1_score_LR > best_LR_metrics.F1_score
        best_LR_metrics.accuracy = current_accuracy_LR;
        best_LR_metrics.precision = current_precision_LR;
        best_LR_metrics.recall = current_recall_LR;
        best_LR_metrics.F1_score = current_F1_score_LR;

        % Store the best Logistic Regression model
        best_LR_model = mdl_LR;
    end

    % Train Random Forest Classifier
    numTrees = 100; % Number of trees in the forest
    tree = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification');

    % Predict on the test set for Random Forest
    y_pred_RF = predict(tree, X_test);
    y_pred_class_RF = str2double(y_pred_RF);

    % Calculate metrics for Random Forest Classifier evaluation
    current_accuracy_RF = sum(y_pred_class_RF == y_test) / numel(y_test);
    current_precision_RF = sum(y_pred_class_RF & y_test) / sum(y_pred_class_RF);
    current_recall_RF = sum(y_pred_class_RF & y_test) / sum(y_test);
    current_F1_score_RF = 2 * (current_precision_RF * current_recall_RF) / (current_precision_RF + current_recall_RF);

    % Collect metrics for Random Forest Classifier evaluation
    accuracy_vals_RF(fold) = current_accuracy_RF;
    precision_vals_RF(fold) = current_precision_RF;
    recall_vals_RF(fold) = current_recall_RF;
    F1_score_vals_RF(fold) = current_F1_score_RF;

    % Update best Random Forest model if current performance is better
    if current_F1_score_RF > best_RF_metrics.F1_score
        best_RF_metrics.accuracy = current_accuracy_RF;
        best_RF_metrics.precision = current_precision_RF;
        best_RF_metrics.recall = current_recall_RF;
        best_RF_metrics.F1_score = current_F1_score_RF;

        % Store the best Random Forest model
        best_RF_model = tree;
    end
end


% Calculate average performance metrics across folds for Logistic Regression
avg_accuracy_LR = mean(accuracy_vals_LR);
avg_precision_LR = mean(precision_vals_LR);
avg_recall_LR = mean(recall_vals_LR);
avg_F1_score_LR = mean(F1_score_vals_LR);

% Calculate average performance metrics across folds for Random Forest
avg_accuracy_RF = mean(accuracy_vals_RF);
avg_precision_RF = mean(precision_vals_RF);
avg_recall_RF = mean(recall_vals_RF);
avg_F1_score_RF = mean(F1_score_vals_RF);

% Plotting training and testing performance for Logistic Regression and
% Random Forest 
figure;
subplot(2, 1, 1);
plot(1:k, accuracy_vals_LR, 'bo-', 'LineWidth', 1.5);
hold on;
plot(1:k, precision_vals_LR, 'go-', 'LineWidth', 1.5);
plot(1:k, recall_vals_LR, 'ro-', 'LineWidth', 1.5);
plot(1:k, F1_score_vals_LR, 'co-', 'LineWidth', 1.5);
hold off;
xlabel('Fold');
ylabel('Performance');
title('Logistic Regression: Performance across Folds');
legend('Accuracy', 'Precision', 'Recall', 'F1 Score', 'Location', 'best');
grid on;

subplot(2, 1, 2);
plot(1:k, accuracy_vals_RF, 'bo-', 'LineWidth', 1.5);
hold on;
plot(1:k, precision_vals_RF, 'go-', 'LineWidth', 1.5);
plot(1:k, recall_vals_RF, 'ro-', 'LineWidth', 1.5);
plot(1:k, F1_score_vals_RF, 'co-', 'LineWidth', 1.5);
hold off;
xlabel('Fold');
ylabel('Performance');
title('Random Forest: Performance across Folds');
legend('Accuracy', 'Precision', 'Recall', 'F1 Score', 'Location', 'best');
grid on;

% Displaying average metrics for Logistic Regression and Random Forest
disp('Average Metrics across Folds for Logistic Regression:');
disp(['Average Accuracy: ', num2str(avg_accuracy_LR)]);
disp(['Average Precision: ', num2str(avg_precision_LR)]);
disp(['Average Recall: ', num2str(avg_recall_LR)]);
disp(['Average F1 Score: ', num2str(avg_F1_score_LR)]);
disp(' ')

disp('Average Metrics across Folds for Random Forest Classifier:');
disp(['Average Accuracy: ', num2str(avg_accuracy_RF)]);
disp(['Average Precision: ', num2str(avg_precision_RF)]);
disp(['Average Recall: ', num2str(avg_recall_RF)]);
disp(['Average F1 Score: ', num2str(avg_F1_score_RF)]);
disp(' ')

% Display metrics of the best Logistic Regression model
disp('Metrics of the Best Logistic Regression Model:');
disp(['Accuracy: ', num2str(best_LR_metrics.accuracy)]);
disp(['Precision: ', num2str(best_LR_metrics.precision)]);
disp(['Recall: ', num2str(best_LR_metrics.recall)]);
disp(['F1 Score: ', num2str(best_LR_metrics.F1_score)]);
disp(' ')

% Display metrics of the best Random Forest model
disp('Metrics of the Best Random Forest Model:');
disp(['Accuracy: ', num2str(best_RF_metrics.accuracy)]);
disp(['Precision: ', num2str(best_RF_metrics.precision)]);
disp(['Recall: ', num2str(best_RF_metrics.recall)]);
disp(['F1 Score: ', num2str(best_RF_metrics.F1_score)]);

% Save the best models (Logistic Regression and Random Forest)
save('best_LR_model.mat', 'best_LR_model');
save('best_RF_model.mat', 'best_RF_model');

% Check if the selected features are part of the models before saving
LR_model_features = mdl_LR.PredictorNames;
RF_model_features = best_RF_model.PredictorNames;

disp('Selected Features:');
disp(selected_features);

disp('Features in Logistic Regression Model:');
disp(LR_model_features);

disp('Features in Random Forest Model:');
disp(RF_model_features);

% Define the file paths to save the CSV files
X_train_file = 'X_train.csv';
y_train_file = 'y_train.csv';
X_test_file = 'X_test.csv';
y_test_file = 'y_test.csv';

% Save X_train as CSV
writematrix(X_train, X_train_file);

% Save y_train as CSV
writematrix(y_train, y_train_file);

% Save X_test as CSV
writematrix(X_test, X_test_file);

% Save y_test as CSV
writematrix(y_test, y_test_file);

% Predict on the whole dataset for Logistic Regression
y_pred_LR_whole = predict(best_LR_model, new_X);

% Convert predictions to binary classes (0 or 1)
y_pred_class_LR_whole = round(y_pred_LR_whole);

% Create a confusion matrix for Logistic Regression
confusion_LR = confusionmat(new_y, y_pred_class_LR_whole);

% Visualize the confusion matrix for Logistic Regression
figure;
confusionchart(confusion_LR, {'Not Bankrupt', 'Bankrupt'}, 'Title', 'Confusion Matrix for Logistic Regression');

% Predict on the whole dataset for Random Forest
y_pred_RF_whole = predict(best_RF_model, new_X);

% Convert predictions to binary classes (0 or 1)
y_pred_class_RF_whole = round(str2double(y_pred_RF_whole));

% Create a confusion matrix for Random Forest
confusion_RF = confusionmat(new_y, y_pred_class_RF_whole);

% Visualize the confusion matrix for Random Forest
figure;
confusionchart(confusion_RF, {'Not Bankrupt', 'Bankrupt'}, 'Title', 'Confusion Matrix for Random Forest');

