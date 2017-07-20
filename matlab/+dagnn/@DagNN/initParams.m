function initParams(obj)
% INITPARAM  Initialize the paramers of the DagNN
%   OBJ.INITPARAM() uses the INIT() method of each layer to initialize
%   the corresponding parameters (usually randomly).

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

for l = 1:numel(obj.layers)
    p = obj.getParamIndex(obj.layers(l).params) ;
    %params = obj.layers(l).block.initParams() ;
    params=[];
    if(isequal(class(obj.layers(l).block),'dagnn.Conv'))
        size=obj.layers(l).block.size;
        h =size(1); w =size(2); in = size(3); out=size(4);
        sc = sqrt(3/(h*w*in)) ;
        params{1,1} = (rand(h, w, in, out, 'single')*2 - 1)*sc;
        %sc = sqrt(2/(h*w*out)) ;
        %params{1,1}= randn(h, w, in, out, 'single')*sc ;
        if(obj.layers(l).block.hasBias)
            params{1,2} = zeros(out, 1, 'single') ;
        end
    elseif(isequal(class(obj.layers(l).block),'dagnn.ConvTranspose'))
        sizet=obj.layers(l).block.size;
        h =sizet(1); w =sizet(2); in = sizet(3); out=sizet(4);
        sc = sqrt(3/(h*w*in)) ;
        params{1,1} = (rand(h, w, in, out, 'single')*2 - 1)*sc;
        %sc = sqrt(2/(h*w*in)) ;
        %params{1,1}= randn(h, w, in, out, 'single')*sc ;
        %params{1,1} = bilinear_u(h,in,out);
        params{1,2} = zeros(in, 1, 'single') ;
    elseif(isequal(class(obj.layers(l).block),'dagnn.BatchNorm'))
        %size=obj.layers(l).block.size;
        %h =size(1); w =size(2); in = size(3); out=size(4);
        in = 4096;
        params{1,1} = ones(in,1, 'single') ;
        params{1,2} = zeros(in, 1, 'single') ;
        params{1,3} = zeros(in, 2, 'single') ;
    else
        params = obj.layers(l).block.initParams() ;
    end
    switch obj.device
        case 'cpu'
            params = cellfun(@gather, params, 'UniformOutput', false) ;
        case 'gpu'
            params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
    end
    pppp = isequal(class(obj.layers(l).block),'dagnn.Conv') || ...
        isequal(class(obj.layers(l).block),'dagnn.BatchNorm') ||...
        isequal(class(obj.layers(l).block),'dagnn.ConvTranspose');
    if(pppp&&~isempty(obj.params(p(1)).value))  %for fintune
       continue; 
    end
    [obj.params(p).value] = deal(params{:}) ;
    if(isequal(class(obj.layers(l).block),'dagnn.Conv'))
        [obj.params(p(1)).learningRate]=0.01;
        [obj.params(p(1)).trainMethod] = 'gradient';
        if(obj.layers(l).block.hasBias)
            [obj.params(p(2)).learningRate]=0.2;
            [obj.params(p(2)).trainMethod] = 'gradient';
        end
        if(l<numel(obj.layers) && ~isempty(strfind(class(obj.layers(l+1).block),'Loss')))
            [obj.params(p(1)).learningRate]= 0.01;
            if(obj.layers(l).block.hasBias)
                [obj.params(p(2)).learningRate]= 0.2;
            end
            obj.params(p(1)).value = obj.params(p(1)).value * 0.1;
        end
    end
    if(isequal(class(obj.layers(l).block),'dagnn.BatchNorm'))
        [obj.params(p(1)).learningRate]=2;
        [obj.params(p(2)).learningRate]=1;
        [obj.params(p(3)).learningRate]=0.5;
        [obj.params(p(1)).weightDecay]=0;
        [obj.params(p(2)).weightDecay]=0;
        [obj.params(p(3)).weightDecay]=0;
    end
end
