%% (mini-batch) Stochastic variance reduced gradient (SVRG)
function [cost_vs_iter, step_vs_iter, norm_grad1_vs_iter] = stochastic_variance_reduced_gradient(X, y, N, algo_struct)

%w_init                 = algo_struct.w_init;
w3_init                 = algo_struct.w3_init;
W2_init                 = algo_struct.W2_init;
W1_init                 = algo_struct.W1_init;
cost_func_handle       = algo_struct.cost_func_handle;
%grad_handle            = algo_struct.grad_handle;
%grad_per_sample_handle = algo_struct.grad_per_sample_handle;
grad_J_cost_L2_w3       = algo_struct.grad_w3_handle;
grad_J_cost_L2_W2       = algo_struct.grad_W2_handle;
grad_J_cost_L2_W1       = algo_struct.grad_W1_handle;
nrof_iter              = algo_struct.nrof_iter;
nrof_iter_inner_loop   = algo_struct.nrof_iter_inner_loop; % EPOCH LENGTH
step_size              = algo_struct.step_size; % fixed value is used
step_size_method       = algo_struct.step_size_method;
mini_batch_size        = algo_struct.mini_batch_size; % mini_batch_size==1 => SGD ... mini_batch_size > 1 => mini_batch SGD
mini_batch_rng_gen     = algo_struct.mini_batch_rng_gen; % random number
lambda                 = algo_struct.lambda_reg;

rng(mini_batch_rng_gen);

% w_vs_iter            = zeros(numel(w_init), nrof_iter+1);
% w_vs_iter(:,1)       = w_init;

step_vs_iter         = zeros(nrof_iter+1, 1);
step_vs_iter(1)      = step_size;

cost_vs_iter         = ones(nrof_iter+1, 1); % +1 for initialization
cost_vs_iter(1)      = cost_func_handle(X.', y.', N, W1_init, W2_init, w3_init);

%norm_grad_vs_iter = zeros(nrof_iter+1, 1);
norm_grad3_vs_iter = zeros(nrof_iter+1,1);
norm_grad2_vs_iter = zeros(nrof_iter+1,1);
norm_grad1_vs_iter = zeros(nrof_iter+1,1);

%w_svrg               = w_init;
w3_svrg    = w3_init;              
W2_svrg    = W2_init;
W1_svrg    = W1_init;
d = size(X,2);
step_alpha           = step_size; % initial
counter              = 0;
for kk_outer = 1:nrof_iter % outer-loop /nr of epochs
               
    
    grad_w3 = zeros(size(w3_init));
    grad_W2 = zeros(size(W2_init));
    grad_W1 = zeros(size(W1_init));

    % gradient delta tilda f
    for kk_N = 1:N
        grad_w3 = grad_w3 + grad_J_cost_L2_w3(X(kk_N,:).', y(kk_N),N, W1_svrg , W2_svrg,w3_svrg);
        grad_W2 = grad_W2 + grad_J_cost_L2_W2(X(kk_N,:).', y(kk_N),N, W1_svrg , W2_svrg,w3_svrg);
        grad_W1 = grad_W1 + grad_J_cost_L2_W1(X(kk_N,:).', y(kk_N),N, W1_svrg , W2_svrg,w3_svrg);
    end
    %W_k
    w3_K       = w3_svrg;
    W2_K       = W2_svrg;
    W1_K       = W1_svrg;
   
    cost_vs_iter(kk_outer+1) = cost_func_handle(X.', y.', N, W1_svrg , W2_svrg,w3_svrg);
        
    for tt_inner = 1:nrof_iter_inner_loop
        
        [perm_indices]     = randperm(N,mini_batch_size);
        X_mini_batch       = X(perm_indices,:);
        y_mini_batch       = y(perm_indices);
        
        counter            = counter + 1;
        switch lower(step_size_method)
            case 'fixed'
                step_alpha = step_size;
            case {'decay'; 'decay1'}
                step_alpha = step_size / (1 + tt_inner);
            case 'adaptive' % different from other methods
                step_alpha = step_size / (1 + step_size * lambda * counter);
            case 'adaptive_bb'
                if tt_inner > 1
                    delta_grad  = grad_w_current_svrg - grad_w_prev_svrg;
                    delta_w     = w_svrg - w_prev_svrg;
                    step_alpha = compute_step_size__barzilai_borwein_method(delta_w, delta_grad);
                    if isnan(step_alpha) || (step_alpha==inf) || (step_alpha<=0)
                        step_alpha = step_size; % better be safe
                    end
                else
                    step_alpha = step_size;
                end
            otherwise
                error('unknown step size computation method');
        end
        
        % calculate gradient
        T_grad_w3 = zeros(size(w3_init));
        T_grad_W2 = zeros(size(W2_init));
        T_grad_W1 = zeros(size(W1_init));
        
        K_grad_w3 = zeros(size(w3_init));
        K_grad_W2 = zeros(size(W2_init));
        K_grad_W1 = zeros(size(W1_init));
        
        for kk_mini = 1:mini_batch_size
        kk_N = perm_indices(kk_mini);
        T_grad_w3 = T_grad_w3 + grad_J_cost_L2_w3(X(kk_N,:).', y(kk_N),mini_batch_size, W1_svrg , W2_svrg,w3_svrg);
        T_grad_W2 = T_grad_W2 + grad_J_cost_L2_W2(X(kk_N,:).', y(kk_N),mini_batch_size, W1_svrg , W2_svrg,w3_svrg);
        T_grad_W1 = T_grad_W1 + grad_J_cost_L2_W1(X(kk_N,:).', y(kk_N),mini_batch_size, W1_svrg , W2_svrg,w3_svrg);
        
        K_grad_w3 = K_grad_w3 + grad_J_cost_L2_w3(X(kk_N,:).', y(kk_N),mini_batch_size, W1_K , W2_K,w3_K);
        K_grad_W2 = K_grad_W2 + grad_J_cost_L2_W2(X(kk_N,:).', y(kk_N),mini_batch_size, W1_K , W2_K,w3_K);
        K_grad_W1 = K_grad_W1 + grad_J_cost_L2_W1(X(kk_N,:).', y(kk_N),mini_batch_size, W1_K , W2_K,w3_K);
        
        end
        
        
        % update weights
        W1_svrg = W1_svrg - step_alpha * (T_grad_W1 - K_grad_W1 + grad_W1);
        W2_svrg = W2_svrg - step_alpha * (T_grad_W2 - K_grad_W2 + grad_W2);
        w3_svrg = w3_svrg - step_alpha * (T_grad_w3 - K_grad_w3 + grad_w3);
    end
        cost_vs_iter(kk_outer+1) = cost_func_handle(X.', y.', N, W1_svrg , W2_svrg,w3_svrg);

        %cost_vs_iter(kk_outer+1) = cost_func_handle(X, y, N, w_svrg, lambda);
        
        step_vs_iter(kk_outer+1) = step_alpha;
        norm_grad3_vs_iter(kk_outer+1) = norm(grad_w3,2);
        norm_grad2_vs_iter(kk_outer+1) = norm(grad_W2,2);
        norm_grad1_vs_iter(kk_outer+1) = norm(grad_W1,2); 
        
        
    
end

% Saving data
% (mini-batch) SVRG
str_mbSVRG = strcat('CA3_results/mbSVRG_',algo_struct.alpha_str,'_Lambda',num2str(lambda),...
            '_BatchS',num2str(mini_batch_size),'_Epoch',num2str(nrof_iter_inner_loop));
save(strcat(str_mbSVRG,'.mat'),'cost_vs_iter','step_vs_iter',...
    'norm_grad1_vs_iter','norm_grad2_vs_iter','norm_grad3_vs_iter');

end