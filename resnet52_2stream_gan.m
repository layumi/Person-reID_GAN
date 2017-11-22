function nn = resnet52_2stream()
% Concat two model.
if(~exist('net.mat'))
    net1 = resnet52_market();
    net1.removeLayer('top5err');
    net2 = resnet52_market(); %imagenet
    net2.removeLayer('top5err');
    %change name
    for i = 1:numel(net2.layers)
        net2.renameLayer(net2.layers(i).name,sprintf('%s_2',net2.layers(i).name));
    end
    for i = 1:numel(net2.vars)
        net2.renameVar(net2.vars(i).name,sprintf('%s_2',net2.vars(i).name));
    end
    nn = concat_2net(net1,net2);
    net_struct = nn.saveobj();
    save('net.mat','net_struct');
else
    load('net.mat');
    nn = dagnn.DagNN.loadobj(net_struct);
end
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



