function [W, b, grad_struct, E_abs_square_avg, avg_index] ...
    = decentralized_optimization__weights_bias__dnn(X_train, Y_train, n_train, mini_batch_size, mini_batch_rng_gen, ...
    X_test,y_test,n_test,...
    W, b, bias_enable_flag, batch_norm_flag, ...
    z, z_prime, h,...
    a_fun, da_fun, hyperparameters_activation_function, ...
    select_optimization_method, step_struct, step_size_method, ...
    regularization_factor, regularization_type,...
    nrof_nodes_output,nrof_nodes_vec,nrof_hidden_layers,nrof_total_layers,...
    nrof_epochs, epoch_count, ...
    E_abs_square_avg, avg_index, enable_online_plotting)

% Initialize
if epoch_count==1
    res.E_abs_square_avg_over_epoch         = zeros(nrof_epochs, nrof_nodes_output);
    res.classification_accuracy_over_epoch  = zeros(nrof_epochs, 1);
end

%
for lyr = 1:nrof_hidden_layers
    z{lyr}           = zeros(nrof_nodes_vec(lyr+1),mini_batch_size); % output after activation
    h{lyr}           = zeros(nrof_nodes_vec(lyr+1),mini_batch_size); % output before activation
    z_prime{lyr}     = ones(nrof_nodes_vec(lyr+1),mini_batch_size); % output with gradient of the activation
    %diagZ_prime{lyr} = zeros(nrof_nodes_vec(lyr+1), nrof_nodes_vec(lyr+1), mini_batch_size); % output with gradient of the activation
end

W_worker = cell(0,nrof_total_layers);
b_worker = cell(0,nrof_total_layers);


%% divide the data set into appropraite nr of mini-batch sizes

rng(mini_batch_rng_gen);

train_data_rand_indices_set = randperm(n_train, n_train);
perm_indices__set           = buffer(train_data_rand_indices_set, mini_batch_size);

nrof_mini_batches           = size(perm_indices__set, 2);
% if epoch_count==1
%     abs_E_n_per_set  = nan(nrof_epochs, nrof_mini_batches, mini_batch_size, nrof_nodes_output);
% end

%% Weights and Biases Update

%update_flag = true; % in case of weights and biases updates
t_outer     = tic;
%profile on;
num_of_workers = 6;
nrof_mini_batches = nrof_mini_batches / num_of_workers;

for t_d = 1:nrof_mini_batches % [parallelize this loop for decentralized set-up]
    for worker_id = 1:num_of_workers
    t = num_of_workers * (t_d - 1) + worker_id;
    %t_inner = tic;
    if mod(t,100)==0; fprintf('nr of minibatches so far=%d \n',t); end
    
    % randomly select the mini-batch sizes for the training ...
    % (should be non-overlapping)
    [perm_indices]      = perm_indices__set(:, t);
    X_train__mini_batch = X_train(:,perm_indices);
    Y_train__mini_batch = Y_train(:,perm_indices);
    
    % initialize
    Y_pred__mini_batch  = nan(size(Y_train__mini_batch));
    
    % PREDICTION (feedforward)
    [Y_pred__mini_batch(:,:), h, z, z_prime] = dnn__prediction(X_train__mini_batch, W, b, bias_enable_flag, mini_batch_size, nrof_total_layers, batch_norm_flag, ...
        h, z, z_prime, ...
        a_fun, da_fun, hyperparameters_activation_function);
    
    % =================================================================
    % error to be utilized for updating Weights & Biases (parameters)
    % =================================================================
    % compute output layer error: (Y - T)
    E_n  = Y_pred__mini_batch(:,:) - Y_train__mini_batch(:,:);
    
    
    % logger (for the cost computation)
    %abs_E_n_per_set(epoch_count, t,:,:) = abs(E_n).';
    [E_abs_square_avg, avg_index]       = recursive_average(E_abs_square_avg, mean(abs(E_n).^2,2), avg_index);
    
    % =================================================================
    % Update the weights and biases (after gradient computation--back propagation)
    % =================================================================
    grad_struct = compute_gradients_for_deep_neural_network_ver2(E_n, X_train__mini_batch, W, b, z, z_prime, mini_batch_size,...
        nrof_total_layers, regularization_factor, regularization_type);
    
        
    % Now update the weights and biases
    switch lower(select_optimization_method)
        case 'sgd' % sequential SGD
            kk_outer            = t; % valid for sequential
            [W_temp, b_temp, step_struct] = sgd(W, b, grad_struct, step_struct, step_size_method, kk_outer, nrof_total_layers);
        case 'rmsprop' % averaged version of adagrad
            kk_outer            = t; % valid for sequential
            % If kk_outer == 1, then create the struct that saves gradients
            if avg_index == 1
                grad_prev_struct = struct('layers',{cell(1,nrof_total_layers)});
%                 'dW_storage',[],'db_storage',[]);
            end
            %use here
            [W, b, step_struct, grad_prev_struct] = rmsprop(W, b, grad_struct, grad_prev_struct, step_struct, step_size_method, kk_outer, avg_index, nrof_total_layers);
        case 'adagrad' %
            kk_outer            = t; % valid for sequential
            % If kk_outer == 1, then create the struct that saves gradients
            if kk_outer == 1
                grad_prev_struct = struct('dW_sum',{cell(1,nrof_total_layers)},'db_sum',{cell(1,nrof_total_layers)});
            end
            [W, b, step_struct, grad_prev_struct] = adagrad(W, b, grad_struct, grad_prev_struct, step_struct, step_size_method, kk_outer, nrof_total_layers);
        case 'adam'
            error('not supported yet');
    end
    
    % just for debugging... online plotting: caveat --> slow
    switch 1
        case enable_online_plotting.inner_loop==1 &&  mod(t,100)==0
            figure(1);
            switch 1
                case (avg_index==1)
                    clf;
            end
            semilogy(avg_index, E_abs_square_avg, 'o');
            hold on;
            drawnow;
            xlabel('nr of mini-batches');
            ylabel('mse: |y - t|^2');
    end
    
    %toc(t_inner);
    
    W_worker = W_worker + W_temp;
    b_worker = b_worker + b_temp;
    
    end % worker loop end
    
    W = W_worker / num_of_workers;
    b = b_worker / num_of_workers;
    W_worker = cell(0,nrof_total_layers);
    b_worker = cell(0,nrof_total_layers);
    
    %profview;
    % ==========================
    % Save the result every t minibatches
    % ==========================
    if mod(t,100)==0
        res.W    = W;
        res.b    = b;
        res.grad = grad_struct;
        res.E_abs_square_avg = E_abs_square_avg;
        
        save(sprintf('result/interim__cent_%s_%dmini_biasEn%d__%dlyr_dnn.mat',...
            select_optimization_method, mini_batch_size, ...
            bias_enable_flag, nrof_total_layers), '-struct', 'res');
    end
    
end % of t: mini batches
toc(t_outer);

% just for debugging... online plotting: caveat --> slow
switch 1
    case enable_online_plotting.outer_loop==1
        figure(2);
        switch 1
            case (epoch_count==1)
                clf;
        end
        semilogy(epoch_count, E_abs_square_avg, 'o');
        %semilogy(epoch_count, squeeze(mean(mean(abs_E_n_per_set(epoch_count,:,:,:),3),2)).^2, 'o');
        hold on;
        drawnow;
        xlabel('nr of epochs');
        ylabel('mse: |y - t|^2');
end

%% Validate the result with the updated weights and biases

%update_flag = false; % in case of weights and biases updates

% flush these intermediate variables
for lyr = 1:nrof_hidden_layers
    z{lyr}           = zeros(nrof_nodes_vec(lyr+1),n_test); % output after activation
    h{lyr}           = zeros(nrof_nodes_vec(lyr+1),n_test); % output before activation
    z_prime{lyr}     = ones(nrof_nodes_vec(lyr+1),n_test); % output with gradient of the activation
    %diagZ_prime{lyr} = zeros(nrof_nodes_vec(lyr+1), nrof_nodes_vec(lyr+1), n_test); % output with gradient of the activation
end

[Y_pred__test_data] = dnn__prediction(X_test, W, b, bias_enable_flag, size(X_test,2), nrof_total_layers, batch_norm_flag, ...
    h, z, z_prime,  ...
    a_fun, da_fun, hyperparameters_activation_function);
[val, y_pred__final]    = max(Y_pred__test_data, [], 1);
y_pred__final           = (y_pred__final-1).';
[ind, val_]             = find(y_pred__final==y_test);
classification_accuracy = (numel(ind)/numel(y_test))*100;
fprintf('classification accuracy (%d epoch) = %1.2f\n', epoch_count, classification_accuracy);

%% Save the result every epoch
res.W                       = W;
res.b                       = b;
res.grad                    = grad_struct;
res.E_abs_square_avg        = E_abs_square_avg;
res.E_abs_square_avg_over_epoch(epoch_count,:)      = E_abs_square_avg;
res.classification_accuracy_over_epoch(epoch_count) = classification_accuracy;

save(sprintf('result/cent_%s_%dmini_biasEn%d__%dlyr_dnn.mat',...
    select_optimization_method, mini_batch_size, ...
    bias_enable_flag, nrof_total_layers), '-struct', 'res');

% Save previous written information in a file
file_name = strcat('Display_Information_',select_optimization_method,'.txt');
fid=fopen(file_name,'a');

fprintf(fid,'Time elapsed (%d epoch) in seconds = %1.2f\n',epoch_count,t_outer);
fprintf(fid,'classification accuracy (%d epoch) = %1.2f\n\n', epoch_count, classification_accuracy);

% Close the file
fclose(fid);

end
%%
