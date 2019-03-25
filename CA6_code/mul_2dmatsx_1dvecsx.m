%%  matrix  vector multiplication with singleton expansion: C(:,i1,i2,..,iN) = A(:,:,i1,i2,...,iN)*b(:,i1,i2,...,iN)
function C = mul_2dmatsx_1dvecsx(A,b)
    nrof_dims_b = length(size(b));
    perm_b = [1 nrof_dims_b+1 2:nrof_dims_b];
    C = ipermute(mul_2dmatsx_2dmatsx(A, permute(b, perm_b)),perm_b);
end