addpath(genpath('./multifocal'))

path_results = './prediction2fuse_multifocal';

channel = 'average';
% fold = 1;
for fold = 1:4
    filename_mat = strcat('results_fold',num2str(fold),'_',channel,'.mat');
    filename_full = fullfile(path_results, filename_mat);

    load(filename_full);

    xgb_train = log(xgb_train)';
    cnns_train = log(cnns_train)';
    y_train = double(y_train + 1);
    [alpha,beta] = train_nary_llr_fusion({xgb_train,cnns_train},y_train);

    xgb_test = log(xgb_test)';
    cnns_test = log(cnns_test)';
    pred_proba_fusion = apply_nary_lin_fusion({xgb_test,cnns_test},alpha,beta);

    [~, y_fusion] = max(pred_proba_fusion,[],1);
    y_fusion = y_fusion - 1;

    save(fullfile(path_results,strcat('y_fusion_fold',num2str(fold),'_',channel)), 'y_fusion');
end