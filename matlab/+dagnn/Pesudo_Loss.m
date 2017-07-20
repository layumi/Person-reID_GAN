classdef Pesudo_Loss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      labels = inputs{2};
      labels_new = update_pesudo(inputs{1},labels);
      instanceWeights = ones(size(labels));
      gan = (labels==0);
      instanceWeights(gan) = 0.1;
      instanceWeights = reshape(instanceWeights,1,1,1,[]);
      outputs{1} = vl_nnloss(inputs{1},labels_new, [], 'loss', obj.loss,'instanceWeights',instanceWeights) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      labels = inputs{2};
      labels_new = update_pesudo(inputs{1},labels);
      instanceWeights = ones(size(labels));
      gan = (labels==0);
      instanceWeights(gan) = 0.1;
      instanceWeights = reshape(instanceWeights,1,1,1,[]);
      derInputs{1} = vl_nnloss(inputs{1}, labels_new, derOutputs{1}, 'loss', obj.loss, 'instanceWeights',instanceWeights,obj.opts{:}) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
