classdef ODistLoss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      result = abs(bsxfun(@minus,inputs{1},inputs{2}));
      result(result<1) =  result(result<1).^2;
      outputs{1} = sum(result(:));
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      bp = bsxfun(@minus,inputs{1},inputs{2});
      bp(bp<-1) = -0.5;
      bp(bp>1) = 0.5;
      derInputs{1} = 2.*bsxfun(@times,bp,derOutputs{1});
      derInputs{2} = [];
      derParams = {} ;
    end

    function obj = ODistLoss(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
