addpath(genpath('./multifocal'))

path_results = './prediction2fuse_multifocal';

% channel = 'average';
% fold = 1;
for fold = 1:4
    filename_mat_left = strcat('results_rong_test_fold',num2str(fold),'_left.mat');
    filename_mat_right = strcat('results_rong_test_fold',num2str(fold),'_right.mat');
    filename_mat_average = strcat('results_rong_test_fold',num2str(fold),'_average.mat');
    filename_mat_difference = strcat('results_rong_test_fold',num2str(fold),'_difference.mat');

    filename_full = fullfile(path_results, filename_mat_left);
    load(filename_full);
    xgb_train_left = log(xgb_train)';
    cnns_train_left = log(cnns_train)';
    y_train_left = double(y_train + 1);
    xgb_test_left = log(xgb_test)';
    cnns_test_left = log(cnns_test)';
   

    filename_full = fullfile(path_results, filename_mat_right);
    load(filename_full);
    xgb_train_right = log(xgb_train)';
    cnns_train_right = log(cnns_train)';
    y_train_right = double(y_train + 1);
 	xgb_test_right = log(xgb_test)';
    cnns_test_right = log(cnns_test)';
   
	filename_full = fullfile(path_results, filename_mat_average);
    load(filename_full);
    xgb_train_average = log(xgb_train)';
    cnns_train_average = log(cnns_train)';
    y_train_average = double(y_train + 1);
 	xgb_test_average = log(xgb_test)';
    cnns_test_average = log(cnns_test)';
   	
   	filename_full = fullfile(path_results, filename_mat_difference);
    load(filename_full);
    xgb_train_difference = log(xgb_train)';
    cnns_train_difference = log(cnns_train)';
    y_train_difference = double(y_train + 1);
 	xgb_test_difference = log(xgb_test)';
    cnns_test_difference = log(cnns_test)';
   
    [alpha,beta] = train_nary_llr_fusion({xgb_train_left, xgb_train_right, xgb_train_average, xgb_train_difference},y_train_left);

    pred_proba_fusion = apply_nary_lin_fusion({xgb_test_left, xgb_test_right, xgb_test_average, xgb_test_difference},alpha,beta);

    [~, y_fusion] = max(pred_proba_fusion,[],1);
    y_fusion = y_fusion - 1;

    save(fullfile(path_results,strcat('y_fusion_rong_test_gbmfold',num2str(fold))), 'y_fusion');

    [alpha,beta] = train_nary_llr_fusion({cnns_train_left, cnns_train_right, cnns_train_average, cnns_train_difference},y_train_left);

    pred_proba_fusion = apply_nary_lin_fusion({cnns_test_left, cnns_test_right, cnns_test_average, cnns_test_difference},alpha,beta);

    [~, y_fusion] = max(pred_proba_fusion,[],1);
    y_fusion = y_fusion - 1;

    save(fullfile(path_results,strcat('y_fusion_rong_test_cnnsfold',num2str(fold))), 'y_fusion');

	[alpha,beta] = train_nary_llr_fusion({xgb_train_left, xgb_train_right, xgb_train_average, xgb_train_difference, cnns_train_left, cnns_train_right, cnns_train_average, cnns_train_difference},y_train_left);

    pred_proba_fusion = apply_nary_lin_fusion({xgb_test_left, xgb_test_right, xgb_test_average, xgb_test_difference, cnns_test_left, cnns_test_right, cnns_test_average, cnns_test_difference},alpha,beta);

    [~, y_fusion] = max(pred_proba_fusion,[],1);
    y_fusion = y_fusion - 1;

    save(fullfile(path_results,strcat('y_fusion_rong_test_fold',num2str(fold))), 'y_fusion');

end