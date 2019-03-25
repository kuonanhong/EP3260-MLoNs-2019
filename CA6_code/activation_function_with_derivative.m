function [a, da, da1] = activation_function_with_derivative(activation_fun_type)
% Input: activation function type

% Output: function handles of given activation function type
   % a   : activation function handle
   % da  : derivative of the function handle
   % da_ : alternative derivative of the function handle for sanity check

   % [REF] see the list of the activation functions in the below reference --
   % Comparison of new activation functions in neural network for forecasting financial time series

nrof_hidden_layers = numel(activation_fun_type);
a   = cell(1, nrof_hidden_layers+1);
da  = cell(1, nrof_hidden_layers+1);
da1 = cell(1, nrof_hidden_layers+1);
   
for ll = 1:nrof_hidden_layers
    
    switch lower(activation_fun_type{ll})
        case {'relu'}
            a{ll}   = @(z, hyper_par) max(0, z);  % hyper_par is unused
            da{ll}  = @(z, hyper_par) max(sign(z),0) + (z==0).*0.5;  % for z=0, we need to add 0.5 since it is non-differentiable at z=0.
            da1{ll} = da{ll};            
        case {'sigmoid'; 'logsig'}
            a{ll}   = @(z, hyper_par) ( 1 ./ (1 + exp(-z)) );   % hyper_par is unused
            da{ll}  = @(z, hyper_par) ( ( 1 ./ (1 + exp(-z)) )) .* (1 - (1 ./ (1 + exp(-z))) ) ; % == ( exp(-z) ./ ((1 + exp(-z)).^2) ); 
            da1{ll} = @(z) ( exp(-z) ./ ((1 + exp(-z)).^2) ); 
        case {'cloglog'}
            a{ll}   = @(z, hyper_par) ( 1  - exp(-exp(z)) );   % hyper_par is unused
            da{ll}  = @(z, hyper_par) ( 1  - exp(-exp(z)) ) .* exp(z); 
            da1{ll} = da{ll}; 
        case {'cloglogm'}
            a{ll}   = @(z, hyper_par) ( 1  - 2*exp(-0.7*exp(z)) );  % hyper_par is unused
            da{ll}  = @(z, hyper_par) 1.4* exp(-0.7*exp(z)) .* exp(z);
            da1{ll} = da{ll};
        case {'gaussian'; 'se'} % se reads squared-exponential
            a{ll}   = @(z, hyper_par) ( exp(-0.5 * (z.^2) ./ (hyper_par(1).^2)) );
            da{ll}  = @(z, hyper_par) ( exp(-0.5 * (z.^2) ./ (hyper_par(1).^2)) ) .* (-z./(hyper_par(1).^2));
            da1{ll} = da{ll};
        otherwise
            error('unknown activation function types');
    end
end

%% for the output layer, we have an identity mapping
a{ll+1}   = @(z, hyper_par) ( z );
da{ll+1}  = @(z, hyper_par) ( z );
da1{ll+1} = da{ll+1};
