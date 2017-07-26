function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load the dataset
imdb = load('./url_data_gan_6000.mat') ;
imdb = imdb.imdb;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_market_pseudo();
net.conserveMemory = true;
net.meta.normalization.averageImage = reshape([105.6920,99.1345,97.9152],1,1,3);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 64;
opts.train.continue = true; 
opts.train.gpus = 2; %gpu id
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = './data/res52_drop0.75_pesudo1_gan6000' ;
opts.train.derOutputs = {'objective', 1,'objective_pseudo',0} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 5;
opts.train.learningRate = [0.1*ones(1,40),0.01*ones(1,10)] ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag_pseudo(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,net,batch,opts)
% --------------------------------------------------------------------
if(opts.epoch>20)  % after 20 epoch we start using pseudo label.
    opts.derOutputs = {'objective', 0,'objective_pseudo',1} ;
end
im_url = imdb.images.data(batch) ; 
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.85,1],'NumThreads',8);
labels = imdb.images.label(batch) ;
oim = bsxfun(@minus,im{1},opts.averageImage); 
inputs = {'data',gpuArray(oim),'label',single(labels)};
