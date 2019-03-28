% clear variables;

function main_ca6_decentralize(nrof_epochs,opt_meth_number)
addpath 'mtimesx_20110223\'; % "borrowed" a fast multiplication of a multi-dimensional matrices % REF: https://se.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support?s_tid=FX_rc1_behav
% close all;clc;
%%% INPUT
% nrof_epochs           - Number of epochs to train.
% opt_meth_number       - Type of optimization method: 1 is SGD, 2 is Adagrad, and 3 is RMSprop

% =====================================
% Load the training and test MNIST data
% =====================================

% training data
train_x_data_file_name = 'train-images.idx3-ubyte'; % X: INPUT (images)
train_y_data_file_name = 'train-labels.idx1-ubyte'; % y: OUTPUT (label)
[X_train, y_train_]    = load_mnist_data(train_x_data_file_name, train_y_data_file_name);

% test data
test_x_data_file_name = 't10k-images.idx3-ubyte'; % X: INPUT (images)
test_y_data_file_name = 't10k-labels.idx1-ubyte'; % y: OUTPUT (label)
[X_test, y_test]      = load_mnist_data(test_x_data_file_name, test_y_data_file_name);

% derived parameters from the data
n_train            = size(X_train, 2); % total nr of samples
d                  = size(X_train, 1); % dimension of the feature vector
n_test             = size(X_test, 2);  % total nr of samples
d_                 = size(X_test, 1);  % dimension of the feature vector (test) should be same as train
assert(d==d_);

% ########################################################################
%% Some INPUTS
% ########################################################################

% ====================================================
% DNN framework
% ====================================================
nrof_nodes_input     = d;
nrof_nodes_output    = numel(unique(y_train_)); % = 10
nrof_hidden_layers   = 2; % not including the input and output layers
nrof_nodes_hidden    = [700, 700]; %[100, 100, 100, 100, 100, 100, 100]; % the length of this row vector should match with nrof_hidden_layers
assert(length(nrof_nodes_hidden)==nrof_hidden_layers);

% for each hidden layer, we can define the activation functions
activation_functions                = {'sigmoid'; 'sigmoid'}; %{'ReLu'; 'se'; 'ReLu'; 'se'; 'ReLu'; 'se'; 'ReLu'};
hyperparameters_activation_function = [1 , 1]; %[1     ;  1  ;    1  ;   1 ;   1   ;  1  ;   1]; % layer-wise if required. e.g., for SE we need length-scale
assert(numel(activation_functions)==nrof_hidden_layers);

batch_norm_flag                     = false; % TBD
bias_enable_flag                    = false;

% ====================================================
% Optimization for Weights/biases computation
% ====================================================
% nrof_epochs                = 1000; % number of epochs to train.

switch opt_meth_number
    case 1 % SGD
        select_optimization_method = 'SGD'; % 'SGD', 'RMSprop' , 'AdaGrad', 'ADAM'
    case 2 % Adagrad
        select_optimization_method = 'AdaGrad'; % 'SGD', 'RMSprop' , 'AdaGrad', 'ADAM'
    case 3 % RMSprop
        select_optimization_method = 'RMSprop'; % 'SGD', 'RMSprop' , 'AdaGrad', 'ADAM'
    otherwise
        disp('No Algorithm with this number');
end
        
mini_batch_size            = 200; %round(n*10/100); % for mini-batch SGD
mini_batch_rng_gen         = 1256;

regularization_type        = 'l2'; %'l2';
regularization_factor      = 0.01;

step_size_method           = 'fixed';
%for SGD: 'fixed'; 'decay';
%for RMSprop it is adaptive (don't care)
%for AdaGrad it is adaptive (don't care)
%for ADAM it is adaptive    (don't care)
step_size_W_initial        = 0.1; %4e-4; % for all the layers
step_size_b_initial        = 0.1; %4e-4; % for all the layers

% other inputs
enable_online_plotting.inner_loop = false; % to check the progress
enable_online_plotting.outer_loop = true; % to check the progress
load_previous_weights_biases      = false;

%% derived parameters from the inputs

nrof_total_layers = nrof_hidden_layers+1; % +1 layer extra to account for a layer for input/output per se.
nrof_nodes_vec    = [nrof_nodes_input, nrof_nodes_hidden, nrof_nodes_output]; % length of nodes_vec should be: nrof_total_layers+1

% for training the multioutput DNN,
% create an output matrix (Y_traing): nrof_nodes_output x n_train, such
% that t-th sample in the y_train is mapped as 1 in the respective Y_train
% matrix
Y_train            = zeros(nrof_nodes_output, n_train);
for tt = 1:n_train
    Y_train(y_train_(tt)+1,tt) = 1;
end

% obtain the activation function handles for the hidden layers
[a_fun, da_fun] = activation_function_with_derivative(activation_functions);

%% Initialization of Deep (fully-connected) neural network (DNN)

% weights
W = cell(1,nrof_total_layers);
% biases
b = cell(1,nrof_total_layers);
% intermediate/hidden outputs
z = cell(1,nrof_total_layers);
% activation outputs (from hidden nodes)
h = cell(1,nrof_total_layers);
% used for the gradients computation
z_prime     = cell(1,nrof_total_layers);
%diagZ_prime = cell(1,nrof_total_layers);

% NN warm-up with the random weights and biases for both input and the hidden layers
for lyr = 1:nrof_total_layers % +1 layers between input and output
    %    nrof_nodes_hidden x nrof_nodes_input [Nr x Nt] (channel model style)
    % or nrof_nodes_output x nrof_nodes_hidden
%     if(isfile(sprintf('result/cent_%s_%dmini_biasEn%d__%dlyr_dnn.mat',...
%             select_optimization_method, mini_batch_size, ...
%             bias_enable_flag, nrof_total_layers))) && load_previous_weights_biases
%         % load the previous W and b results rather than starting from the
%         % scratch
%         load(sprintf('result/cent_%s_%dmini_biasEn%d__%dlyr_dnn.mat',...
%             select_optimization_method, mini_batch_size, ...
%             bias_enable_flag, nrof_total_layers));
%         
%     elseif(isfile(sprintf('result/interim__cent_%s_%dmini_biasEn%d__%dlyr_dnn.mat',...
%             select_optimization_method, mini_batch_size, ...
%             bias_enable_flag, nrof_total_layers))) && load_previous_weights_biases
%         % load the previous W and b results rather than starting from the
%         % scratch
%         load(sprintf('result/interim__cent_%s_%dmini_biasEn%d__%dlyr_dnn.mat',...
%             select_optimization_method, mini_batch_size, ...
%             bias_enable_flag, nrof_total_layers));
%     else
        W{lyr} = rand(nrof_nodes_vec(lyr+1),nrof_nodes_vec(lyr)) - 0.5;
        b{lyr} = rand(nrof_nodes_vec(lyr+1),1) - 0.5;
% s    end
    
    step_struct.W(lyr).step_size = step_size_W_initial; % initial fixed step-size for the weight
    step_struct.b(lyr).step_size = step_size_b_initial; % initial fixed step-size for the biases
    step_struct.lambda           = regularization_factor;
end



%Y_pred_mat      = zeros(nrof_nodes_output, n_train);
%abs_E_avg       = 0;




% #########################################################################
%% DNN training mode
% #########################################################################

centralized__E_abs_square_avg = zeros(nrof_nodes_output, 1);
centralized__avg_index        = 0;

for epoch_count = 1: nrof_epochs
    
    if mod(epoch_count,100)==0; fprintf('nr of epochs so far=%d \n',epoch_count); end
    
    
    % centralized optimization methods
    [W, b, grad_struct,centralized__E_abs_square_avg, centralized__avg_index] ...
        = decentralized_optimization__weights_bias__dnn(X_train, Y_train, n_train, mini_batch_size, mini_batch_rng_gen+(2*(epoch_count-1)), ...
        X_test,y_test,n_test,...
        W, b, bias_enable_flag, batch_norm_flag, ...
        z, z_prime, h,...
        a_fun, da_fun, hyperparameters_activation_function, ...
        select_optimization_method, step_struct, step_size_method, ...
        regularization_factor, regularization_type,...
        nrof_nodes_output,nrof_nodes_vec,nrof_hidden_layers,nrof_total_layers,...
        nrof_epochs, epoch_count, ...
        centralized__E_abs_square_avg, centralized__avg_index, enable_online_plotting);
    
    
end


end


