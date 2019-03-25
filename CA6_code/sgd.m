%% (mini-batch) Stochastic Gradient Descent
function [W, b, step_struct] = sgd(W, b, grad_struct, step_struct, step_size_method, kk_outer, nrof_total_layers)

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
       
    % Update Weights and biases
    W{lyr}  = W_prev - step_struct.W(lyr).step_size * dW_prev;
    
    b{lyr}  = b_prev - step_struct.W(lyr).step_size * db_prev;
   
    % % Saving data
    % % Gradient Descent
    % str_GD = strcat('CA2_results/fullGD_',algo_struct.alpha_str,'_Lambda',num2str(lambda));
    % save(strcat(str_GD,'.mat'),'w_vs_iter','cost_vs_iter','step_vs_iter',...
    %     'norm_grad_vs_iter');
    
end