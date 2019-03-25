function [Y_pred__mini_batch,h, z, z_prime] = dnn__prediction(X_in, W, b, bias_enable_flag, mini_batch_size, nrof_total_layers, batch_norm_flag, ...
                                                                          h, z, z_prime, ...
                                                                          a_fun, da_fun, hyperparameters_activation_function)
% =================================================================
% feedforward (or prediction) mode
% =================================================================
% predict output y given input x based on the current weights
for lyr = 1:nrof_total_layers
    if lyr < nrof_total_layers
        hyper_par  = hyperparameters_activation_function(lyr);
    end
    switch 1
        case lyr==1
            % for the first layer: mapping from input layer -> hidden layer
            % perform: activation(W*x + b)            
            %C            = mul_2dmatsx_1dvecsx(W{lyr},X_in); % linear transformation
            C            = W{lyr} * X_in; 
            %tic; C            = mtimesx(W{lyr},X_in); toc; % linear transformation
            
            h{lyr}       = C + bias_enable_flag * repmat(b{lyr}, [1, mini_batch_size]); % apply biases
            
            z{lyr}       = a_fun{lyr}(h{lyr}, hyper_par);  % apply activation function
            z_prime{lyr} = da_fun{lyr}(h{lyr}, hyper_par); % applying the derivative of the activation function--used for the gradient computation
            
%             if update_flag
%                 % this is really inefficient method
%                 for mm = 1:mini_batch_size
%                     diagZ_prime{lyr}(:,:,mm) = diag(z_prime{lyr}(:,mm));
%                 end
%             end
            
            % for batch normalization
            switch batch_norm_flag==1
                case 1
                    disp('batch norm is NOT done'); % write batch normalization
            end
            
        case lyr==nrof_total_layers
            % for the output layer
            C                   = W{lyr} * z{lyr-1}; %mul_2dmatsx_1dvecsx(W{lyr},z{lyr-1}); % linear transformation
            Y_pred__mini_batch  = C + bias_enable_flag * repmat(b{lyr}, [1, mini_batch_size]); %W{lyr} * z{lyr-1} + (b{lyr});
            
            switch batch_norm_flag==1
                case 1
                    disp('batch norm is NOT done'); % [Question] Do we need batch normalization at the output layer?
            end
            
        otherwise
            % for other hidden layers
            C            = W{lyr} * z{lyr-1}; %mul_2dmatsx_1dvecsx(W{lyr},z{lyr-1}); % linear transformation
            h{lyr}       = C + bias_enable_flag * repmat(b{lyr}, [1, mini_batch_size]);
            
            z{lyr}       = a_fun{lyr}(h{lyr}, hyper_par);
            z_prime{lyr} = da_fun{lyr}(h{lyr}, hyper_par);
            
%             if update_flag
%                 for mm = 1:mini_batch_size
%                     diagZ_prime{lyr}(:,:,mm) = diag(z_prime{lyr}(:,mm));
%                 end
%             end
            
            switch batch_norm_flag==1
                case 1
                    disp('batch norm is NOT done'); % write batch normalization
            end
            
    end
end
end