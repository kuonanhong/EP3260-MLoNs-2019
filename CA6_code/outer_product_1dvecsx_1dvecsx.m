%%  matrix  vector multiplication with singleton expansion: C(:,:,i1,i2,..,iN) = a(:,i1,i2,...,iN) (outer*) transpose(b(:,i1,i2,...,iN))
function C = outer_product_1dvecsx_1dvecsx(a,b)
    nrof_dims_a = length(size(a));
    perm_a      = [1 nrof_dims_a+1 2:nrof_dims_a];

    nrof_dims_b = length(size(b));
    perm_b      = [nrof_dims_b+1 1 2:nrof_dims_b];
   
    %C = ipermute(mul_2dmatsx_2dmatsx(permute(a, perm_a), permute(b, perm_b)),perm_b);
    C = (mul_2dmatsx_2dmatsx(permute(a, perm_a), permute(b, perm_b)));
end