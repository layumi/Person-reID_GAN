function x = getFeature2(net,oim,im_mean,inputname,outputname)
im = bsxfun(@minus,single(oim),im_mean);
net.vars(net.getVarIndex(outputname)).precious = true;
net.eval({inputname,gpuArray(im)}) ;
x = gather(net.vars(net.getVarIndex(outputname)).value);
end

