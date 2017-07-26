function labels = update_pesudo( x,labels)
% quick estimate pesudo label
f = vl_nnsoftmax(x);
f = reshape(f,size(x,3),[]);
[f_max,index] = max(f);
%index(f_max<0.1) = 0;
index = gather(index);
gan = (labels==0);
labels(gan) = index(gan);
end

