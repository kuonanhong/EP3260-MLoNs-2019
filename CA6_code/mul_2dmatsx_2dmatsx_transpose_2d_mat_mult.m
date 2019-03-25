%%  matrix mulitplications with singleton expansion: C(:,:,i1,i2,..,iN) = transpose(A(:,:,i1,i2,...,iN)*B(:,:,i1,i2,...,iN))
function C = mul_2dmatsx_2dmatsx_transpose_2d_mat_mult(A,B)
    first_not_used_dim = max(length(size(A)),length(size(B)))+1;
    permA = 1:first_not_used_dim; permA(2)= first_not_used_dim; permA(first_not_used_dim) = 2;
    permB = 1:first_not_used_dim; permB(1)= first_not_used_dim; permB(first_not_used_dim) = 1;
    C = sum(bsxfun(@times, permute(A, permA), permute(B, permB)), first_not_used_dim);
end