%% (mini-batch) Root Mean Squared step sizes (RMSprop)
function [W, b, step_struct, grad_prev_struct] = rmsprop(W, b, grad_struct, grad_prev_struct, step_struct, step_size_method, kk_outer, avg_index, nrof_total_layers)

% Define epsilon value for avoiding division by zero
epsilon = 1e-8;
% Define gamma as the dampening effect for close and previous iterations
gamma = 0.9;
% Number of accumulated past gradients
omega = 5;

for lyr = 1:nrof_total_layers
    
    % Size of gradient squared matrices to be created
    size_grad_squared = size(grad_struct.W{lyr},1);
    
    switch lower(step_size_method)
        case 'fixed'
            step_struct.W(lyr).step_size = step_struct.W(lyr).step_size;
            step_struct.b(lyr).step_size = step_struct.b(lyr).step_size;
            
        case {'decay'; 'decay1'}
            step_struct.W(lyr).step_size = step_struct.W(lyr).step_size / (1 + kk_outer);
            step_struct.b(lyr).step_size = step_struct.b(lyr).step_size / (1 + kk_outer);
            
        case 'adaptive' % different from other methods
            step_struct.W(lyr).step_size = step_struct.W(lyr).step_size / (1 + step_struct.W(lyr).step_size * step_struct.lambda * kk_outer);
            step_struct.b(lyr).step_size = step_struct.b(lyr).step_size / (1 + step_struct.b(lyr).step_size * step_struct.lambda * kk_outer);
            
        otherwise
            error('unknown step size computation method');
    end
    
    W_prev = W{lyr};
    b_prev = b{lyr};
    
    dW_prev = grad_struct.W{lyr};
    db_prev = grad_struct.b{lyr};
    
    % If initial iteration, consider only current gradient, otherwise get
    % all of them
    if avg_index == 1
        % Save this gradient to be used for the next iterations
        % Evaluating the updated matrix that sums previous gradient outer
        % products - Weights and biases
        dW_update = (1-gamma)*(dW_prev*dW_prev');
        db_update = (1-gamma)*(db_prev*db_prev');
        % Create structs to save data
        grad_prev_struct.layers{lyr}.dW_storage = zeros(size_grad_squared,size_grad_squared,omega);
        grad_prev_struct.layers{lyr}.dW_storage(:,:,avg_index) = dW_prev*dW_prev';
        grad_prev_struct.layers{lyr}.db_storage = zeros(size_grad_squared,size_grad_squared,omega);
        grad_prev_struct.layers{lyr}.db_storage(:,:,avg_index) = db_prev*db_prev';
    else
        if avg_index <= omega
            % Get cells to be averaged
            dW_average = sum(grad_prev_struct.layers{lyr}.dW_storage(:,:,1:avg_index-1),3)/(avg_index-1);
            db_average = sum(grad_prev_struct.layers{lyr}.db_storage(:,:,1:avg_index-1),3)/(avg_index-1);
        else
            % Evaluate average of previous gradients
            dW_average = mean(grad_prev_struct.layers{lyr}.dW_storage,3);
            db_average = mean(grad_prev_struct.layers{lyr}.db_storage,3);
        end
        % Update the iterates
        dW_update = gamma*dW_average + (1-gamma)*(dW_prev*dW_prev');
        db_update = gamma*db_average + (1-gamma)*(db_prev*db_prev');
        % Save the current gradient based on the number of accumulated
        % points
        current_omega = mod(avg_index,omega);
        if current_omega == 0 % this means current_omega should be equal to omega
            current_omega = omega;
        end
        % Evaluating the updated matrix that sums previous gradient outer
        % products - Weights and biases
        grad_prev_struct.layers{lyr}.dW_storage(:,:,current_omega) = dW_prev*dW_prev';
        grad_prev_struct.layers{lyr}.db_storage(:,:,current_omega) = db_prev*db_prev';
        
    end
    
    % Obtain matrix "B = epsilon + sqrt(dW_sum_prev)^-1" for weights and bias
    Bmat_W_condition = epsilon + diag(dW_update);
    Bmat_W_condition = bsxfun(@rdivide, ones(size(Bmat_W_condition)), Bmat_W_condition);
    Bmat_b_condition = epsilon + diag(db_update);
    Bmat_b_condition = bsxfun(@rdivide, ones(size(Bmat_b_condition)), Bmat_b_condition);
    
    % Update Weights and biases
    W{lyr}  = W_prev - step_struct.W(lyr).step_size * diag(Bmat_W_condition) * dW_prev;
    
    b{lyr}  = b_prev - step_struct.W(lyr).step_size * diag(Bmat_b_condition) * db_prev;
    
    % % Saving data
    % % Gradient Descent
    % str_GD = strcat('CA2_results/fullGD_',algo_struct.alpha_str,'_Lambda',num2str(lambda));
    % save(strcat(str_GD,'.mat'),'w_vs_iter','cost_vs_iter','step_vs_iter',...
    %     'norm_grad_vs_iter');
    
end