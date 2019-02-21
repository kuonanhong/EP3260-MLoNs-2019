%% Block coordinate descent
function [ cost_vs_iter, step_vs_iter,norm_grad1_vs_iter] = block_coordinate_descent(X, y, N, algo_struct)

%w_init                 = algo_struct.w_init;
w3_init                 = algo_struct.w3_init;
W2_init                 = algo_struct.W2_init;
W1_init                 = algo_struct.W1_init;
cost_func_handle        = algo_struct.cost_func_handle;
%grad_handle            = algo_struct.grad_handle;
grad_J_cost_L2_w3       = algo_struct.grad_w3_handle;
grad_J_cost_L2_W2       = algo_struct.grad_W2_handle;
grad_J_cost_L2_W1       = algo_struct.grad_W1_handle;
%grad_per_sample_handle = algo_struct.grad_per_sample_handle;
nrof_iter              = algo_struct.nrof_iter;
step_size              = algo_struct.step_size; % fixed value is used
step_size_method       = algo_struct.step_size_method;
lambda                 = algo_struct.lambda_reg;
%step_size_handle  = algo_struct.step_size_handle;

% w3_vs_iter         = zeros(numel(w3_init), nrof_iter+1);
% w3_vs_iter(:,1)    = w_init;

step_vs_iter      = zeros(nrof_iter+1, 1);
step_vs_iter(1)   = step_size;

norm_grad3_vs_iter = zeros(nrof_iter+1,1);
norm_grad2_vs_iter = zeros(nrof_iter+1,1);
norm_grad1_vs_iter = zeros(nrof_iter+1,1);


cost_vs_iter      = ones(nrof_iter+1, 1); % +1 for initialization
cost_vs_iter(1)   = cost_func_handle(X.', y.', N, W1_init, W2_init, w3_init);

%w_gd              = w_init;

w3_bcd    = w3_init;              
W2_bcd    = W2_init;
W1_bcd    = W1_init;

d = size(X,2);

for kk_outer = 1:nrof_iter
    
switch lower(step_size_method)
        case 'fixed'
            step_alpha = step_size;
        case {'decay'; 'decay1'}
            step_alpha = step_size / (1 + kk_outer);
        case 'adaptive' % different from other methods
            step_alpha = step_size / (1 + step_size * 0.01 * kk_outer);
        case 'adaptive_bb'
            if kk_outer > 1
                delta_grad  = grad_w_current - grad_w_prev;
                delta_w     = w_gd - w_gd_prev;
                step_alpha = compute_step_size__barzilai_borwein_method(delta_w, delta_grad);
                if isnan(step_alpha) || (step_alpha==inf) || (step_alpha==0)
                    step_alpha = step_size; % better be safe
                end
            else
                step_alpha = step_size;
            end
        otherwise
            error('unknown step size computation method');
    end
    
    grad_w3 = zeros(size(w3_init));
    grad_W2 = zeros(size(W2_init));
    grad_W1 = zeros(size(W1_init));

    
    for kk_N = 1:N
        grad_w3 = grad_w3 + grad_J_cost_L2_w3(X(kk_N,:).', y(kk_N),N, W1_bcd , W2_bcd,w3_bcd);       
    end
    w3_bcd = w3_bcd - step_alpha*grad_w3;
    
    for kk_N = 1:N
        grad_W2 = grad_W2 + grad_J_cost_L2_W2(X(kk_N,:).', y(kk_N),N, W1_bcd , W2_bcd,w3_bcd);       
    end
    W2_bcd = W2_bcd - step_alpha*grad_W2;
    
    for kk_N = 1:N
        grad_W1 = grad_W1 + grad_J_cost_L2_W1(X(kk_N,:).', y(kk_N),N, W1_bcd , W2_bcd,w3_bcd);
    end
    W1_bcd = W1_bcd - step_alpha*grad_W1;

    
    
    
    %w_vs_iter(:,kk_outer+1)   = w_gd;
    cost_vs_iter(kk_outer+1)  = cost_func_handle(X.', y.', N, W1_bcd , W2_bcd,w3_bcd);
    
    step_vs_iter(kk_outer+1)  = step_alpha;
    
    %norm_grad_vs_iter(kk_outer) = norm(grad_w_prev);
    norm_grad3_vs_iter(kk_outer+1) = norm(grad_w3,2);
    norm_grad2_vs_iter(kk_outer+1) = norm(grad_W2,2);
    norm_grad1_vs_iter(kk_outer+1) = norm(grad_W1,2);    
end

% Saving data
% Gradient Descent
str_BCD = strcat('CA3_results/BCD_',algo_struct.alpha_str,'_Lambda',num2str(lambda));
save(strcat(str_BCD,'.mat'),'cost_vs_iter','step_vs_iter',...
    'norm_grad1_vs_iter','norm_grad2_vs_iter','norm_grad3_vs_iter');

end