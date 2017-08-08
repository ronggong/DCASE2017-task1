addpath(genpath('./multifocal'))

path_results = './prediction2fuse_multifocal';


for fold = 1:4
    filename_mat_developement_train_left = strcat('results_rong_development_train_fold',num2str(fold),'_left.mat');
    filename_mat_developement_train_right = strcat('results_rong_development_train_fold',num2str(fold),'_right.mat');
    filename_mat_developement_train_average = strcat('results_rong_development_train_fold',num2str(fold),'_average.mat');
    filename_mat_developement_train_difference = strcat('results_rong_development_train_fold',num2str(fold),'_difference.mat');

    filename_mat_evaluation_unseen_left = strcat('results_rong_evaluation_unseen_fold',num2str(fold),'_left.mat');
    filename_mat_evaluation_unseen_right = strcat('results_rong_evaluation_unseen_fold',num2str(fold),'_right.mat');
    filename_mat_evaluation_unseen_average = strcat('results_rong_evaluation_unseen_fold',num2str(fold),'_average.mat');
    filename_mat_evaluation_unseen_difference = strcat('results_rong_evaluation_unseen_fold',num2str(fold),'_difference.mat');

    % train data
    filename_full = fullfile(path_results, filename_mat_developement_train_left);
    load(filename_full);
    xgb_train_left = log(xgb_train)';
    cnns_train_left = log(cnns_train)';
    y_train_left = double(y_train + 1);
  

    filename_full = fullfile(path_results, filename_mat_developement_train_right);
    load(filename_full);
    xgb_train_right = log(xgb_train)';
    cnns_train_right = log(cnns_train)';
    y_train_right = double(y_train + 1);
 	
	filename_full = fullfile(path_results, filename_mat_developement_train_average);
    load(filename_full);
    xgb_train_average = log(xgb_train)';
    cnns_train_average = log(cnns_train)';
    y_train_average = double(y_train + 1);
 	
   	filename_full = fullfile(path_results, filename_mat_developement_train_difference);
    load(filename_full);
    xgb_train_difference = log(xgb_train)';
    cnns_train_difference = log(cnns_train)';
    y_train_difference = double(y_train + 1);
 	
    % unseen eval data
    filename_full = fullfile(path_results, filename_mat_evaluation_unseen_left);
    load(filename_full);
    xgb_test_left = log(xgb_test)';
    cnns_test_left = log(cnns_test)';

    filename_full = fullfile(path_results, filename_mat_evaluation_unseen_right);
    load(filename_full);
 	xgb_test_right = log(xgb_test)';
    cnns_test_right = log(cnns_test)';
   
	filename_full = fullfile(path_results, filename_mat_evaluation_unseen_average);
    load(filename_full);
 	xgb_test_average = log(xgb_test)';
    cnns_test_average = log(cnns_test)';
   	
   	filename_full = fullfile(path_results, filename_mat_evaluation_unseen_difference);
    load(filename_full);
 	xgb_test_difference = log(xgb_test)';
    cnns_test_difference = log(cnns_test)';
   
    [alpha,beta] = train_nary_llr_fusion({xgb_train_left, xgb_train_right, xgb_train_average, xgb_train_difference},y_train_left);

    pred_proba_fusion = apply_nary_lin_fusion({xgb_test_left, xgb_test_right, xgb_test_average, xgb_test_difference},alpha,beta);


    save(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_gbmfold',num2str(fold))), 'pred_proba_fusion');

    [alpha,beta] = train_nary_llr_fusion({cnns_train_left, cnns_train_right, cnns_train_average, cnns_train_difference},y_train_left);

    pred_proba_fusion = apply_nary_lin_fusion({cnns_test_left, cnns_test_right, cnns_test_average, cnns_test_difference},alpha,beta);


    save(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_cnnsfold',num2str(fold))), 'pred_proba_fusion');

	[alpha,beta] = train_nary_llr_fusion({xgb_train_left, xgb_train_right, xgb_train_average, xgb_train_difference, cnns_train_left, cnns_train_right, cnns_train_average, cnns_train_difference},y_train_left);

    pred_proba_fusion = apply_nary_lin_fusion({xgb_test_left, xgb_test_right, xgb_test_average, xgb_test_difference, cnns_test_left, cnns_test_right, cnns_test_average, cnns_test_difference},alpha,beta);

    % [~, y_fusion] = max(pred_proba_fusion,[],1);
    % y_fusion = y_fusion - 1;

    save(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_fold',num2str(fold))), 'pred_proba_fusion');

end

load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_gbmfold',num2str(1))));
pred_proba_fusion_gbm_1 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_gbmfold',num2str(2))));
pred_proba_fusion_gbm_2 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_gbmfold',num2str(3))));
pred_proba_fusion_gbm_3 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_gbmfold',num2str(4))));
pred_proba_fusion_gbm_4 = exp(pred_proba_fusion);
pred_proba_fusion_gbm = pred_proba_fusion_gbm_1+pred_proba_fusion_gbm_2+pred_proba_fusion_gbm_3+pred_proba_fusion_gbm_4;

[~, y_fusion] = max(pred_proba_fusion_gbm,[],1);
y_fusion = y_fusion - 1;
save(fullfile(path_results,'y_fusion_rong_evaluation_unseen_gbmfold'), 'y_fusion');

load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_cnnsfold',num2str(1))));
pred_proba_fusion_cnns_1 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_cnnsfold',num2str(2))));
pred_proba_fusion_cnns_2 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_cnnsfold',num2str(3))));
pred_proba_fusion_cnns_3 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_cnnsfold',num2str(4))));
pred_proba_fusion_cnns_4 = exp(pred_proba_fusion);
pred_proba_fusion_cnns = pred_proba_fusion_cnns_1+pred_proba_fusion_cnns_2+pred_proba_fusion_cnns_3+pred_proba_fusion_cnns_4;

[~, y_fusion] = max(pred_proba_fusion_cnns,[],1);
y_fusion = y_fusion - 1;
save(fullfile(path_results,'y_fusion_rong_evaluation_unseen_cnnsfold'), 'y_fusion');

load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_fold',num2str(1))));
pred_proba_fusion_1 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_fold',num2str(2))));
pred_proba_fusion_2 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_fold',num2str(3))));
pred_proba_fusion_3 = exp(pred_proba_fusion);
load(fullfile(path_results,strcat('proba_fusion_rong_evaluation_unseen_fold',num2str(4))));
pred_proba_fusion_4 = exp(pred_proba_fusion);
pred_proba_fusion = pred_proba_fusion_1+pred_proba_fusion_2+pred_proba_fusion_3+pred_proba_fusion_4;

[~, y_fusion] = max(pred_proba_fusion,[],1);
y_fusion = y_fusion - 1;
save(fullfile(path_results,'y_fusion_rong_evaluation_unseen_fold'), 'y_fusion');
