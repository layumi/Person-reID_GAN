function nn = resnet52_2stream()
%This code concat two resnet together
net1 = resnet52_market();
%net1.removeLayer('top1err');
%net1.removeLayer('top5err');
net1 = net1.saveobj() ;
net2 = load('resnet52_market.mat') ;
net1.vars = [net1.vars,net2.vars(2:end)];
net1.layers = [net1.layers,net2.layers];
%net1.meta = [net1.meta,net2.meta];
nn = dagnn.DagNN.loadobj(net1) ;


% *****************************************************************************
% 2 classify
nn.addLayer('Square',dagnn.Square(),{'pool5','pool5_2'},{'ODist'},{});
dropoutBlock = dagnn.DropOut('rate',0.9);
nn.addLayer('dropout_D',dropoutBlock,{'ODist'},{'ODist_d'},{});
fc751Block = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
nn.addLayer('fc9',fc751Block,{'ODist_d'},{'Dist_prediction'},{'fc9f','fc9b'});
lossBlock = dagnn.Loss('loss', 'softmaxlog');
nn.addLayer('softmaxloss_D',lossBlock,{'Dist_prediction','label_f'},'objective_final');
nn.addLayer('top1err_compare', dagnn.Loss('loss', 'classerror'), ...
    {'Dist_prediction','label_f'}, 'top1err_cmopare') ;

nn.layers(174).block.rate = 0.75;
nn.layers(352).block.rate = 0.75;
nn.layers(176).block.loss = 'labelsmooth';
nn.layers(354).block.loss = 'labelsmooth';

nn.initParams();



