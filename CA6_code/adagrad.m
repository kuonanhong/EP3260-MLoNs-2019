%% (mini-batch) Adaptive step sizes (AdaGrad)
function [W, b, step_struct] = adagrad(W, b, grad_struct, step_struct, step_size_method, kk_outer, nrof_total_layers)

% Define epsilon value for avoiding division by zero
epsilon = 1e-8;

for lyr = 1:nrof_total_layers
    
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
    if kk_outer == 1
        % Evaluating the updated matrix that sums previous gradient outer
        % products - Weights and biases
        dW_sum_prev = dW_prev*dW_prev';
        grad_struct.dW_sum_prev{lyr} = dW_sum_prev;
        db_sum_prev = db_prev*db_prev';
        grad_struct.db_sum_prev{lyr} = db_sum_prev;
    else
        % Evaluating the updated matrix that sums previous gradient outer
        % products - Weights and biases
        dW_sum_prev = dW_prev*dW_prev' + grad_struct.dW_sum_prev{lyr};
        grad_struct.dW_sum_prev{lyr} = dW_sum_prev;
        db_sum_prev = db_prev*db_prev' + grad_struct.db_sum_prev;
        grad_struct.db_sum_prev{lyr} = db_sum_prev;
    end
    
    % Obtain matrix "B = epsilon + sqrt(dW_sum_prev)^-1" for weights and bias
    Bmat_W_condition = epsilon + diag(dW_sum_prev);
    Bmat_W_condition = bsxfun(@rdivide, ones(size(Bmat_W_condition)), Bmat_W_condition);
    Bmat_b_condition = epsilon + diag(db_sum_prev);
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