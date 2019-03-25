function [grad] = compute_gradients_for_deep_neural_network_ver2(E_n, X_train__mini_batch, W, b, z, z_prime, mini_batch_size,...
                                                            nrof_total_layers, regularization_factor, regularization_type)

%% 
switch lower(regularization_type)
    case {'none'} % no regularization
        reg_flag = 0;   
    otherwise
        reg_flag = 1;   
end


%% Obtaining gradients (back-propagation)

for lyr = nrof_total_layers:-1:1
    
    switch 1        
            
        case lyr==nrof_total_layers
            % for the output layer  
%             tic
%             E_n__z  = zeros([size(E_n,1), size( z{lyr-1})] );
%             for mm = 1:mini_batch_size
%                 z_per_mb     = z{lyr-1}(:,mm);
%                 E_n_mb         = E_n(:,mm);
%                 E_n__z(:,:,mm) = E_n_mb * z_per_mb.';      
%             end
%             toc
            
            grad.W{lyr} = outer_product_1dvecsx_1dvecsx (E_n , z{lyr-1}) ...
                            + reg_flag * regularization(W{lyr}, regularization_factor, regularization_type);
            
            grad.b{lyr} = E_n ...
                            + reg_flag * regularization(b{lyr}, regularization_factor, regularization_type);
                       
            
        otherwise
            % for first and other hidden layers
            % diagZ_prime_times_W = 1;
            ii = 0;
            for ll = lyr+1:nrof_total_layers % nrof_total_layers:-1:lyr+1
                                
                W_tran       = W{ll}.';
                [~, W_size2] = size(W_tran);
                
%                 tic
%                 M                  = zeros(W_size1, W_size2, mini_batch_size);               
%                 for mm = 1:mini_batch_size
%                     z_prime_per_mb = z_prime{ll-1}(:,mm);
%                     M(:,:,mm)      = repmat(z_prime_per_mb, [1,W_size2]) .*  W_tran;      
%                 end
%                 toc
                               
                %tic %: 
                M  = permute(repmat(z_prime{ll-1}, [1,1,W_size2]),[1,3,2]) .* repmat(W_tran,[1,1,mini_batch_size]); 
                %toc;
                
                ii  = ii + 1;
                if ii==1
                    diagZ_prime_times_W = M;
                else
                    try
                        diagZ_prime_times_W = mtimesx(diagZ_prime_times_W, M);
                    catch
                        diagZ_prime_times_W = mul_2dmatsx_2dmatsx(diagZ_prime_times_W, M);
                    end
                end
            end % of ll loop 
            
            if lyr==1
                C  = outer_product_1dvecsx_1dvecsx(E_n,X_train__mini_batch);
            else
                C  = outer_product_1dvecsx_1dvecsx(E_n,z{lyr-1});
            end
            % transpose 3D matrix
            D  = diagZ_prime_times_W; %permute(diagZ_prime_times_W, [2,1,3]);
            
            try
                grad.W{lyr} = mtimesx(D, C) ...
                                + reg_flag * regularization(W{lyr}, regularization_factor, regularization_type);
            catch
                grad.W{lyr} = mul_2dmatsx_2dmatsx(D, C) ...
                                + reg_flag * regularization(W{lyr}, regularization_factor, regularization_type);
            end
            
            grad.b{lyr} = mul_2dmatsx_1dvecsx(D, E_n) ...
                            + reg_flag * regularization(b{lyr}, regularization_factor, regularization_type);
            
            
    end
    
    % take the mean of the gradients over the (mini-batch) samples
    grad.W{lyr} = mean(grad.W{lyr}, 3);
    grad.b{lyr} = mean(grad.b{lyr}, 2);
end

end

%% for regulrization
function reg = regularization(x, regularization_factor, regularization_type)

switch lower(regularization_type)
    case {'none'} % no regularization
        reg = 0;
    case {'l2'} % L2 regularization
        reg = regularization_factor * x;
        %case {'l1'}
    otherwise
        error('not supported regularization type');
end

end
    