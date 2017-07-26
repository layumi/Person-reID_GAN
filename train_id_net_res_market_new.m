
function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('./url_data.mat') ;
imdb = imdb.imdb;
%imdb.images.set(1:10000) = 3;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_market();

%netStruct = load('/home/zzd/re_ID_beta23_uts0/data/resnet52_2stream_drop0.9_all/net_single.mat') ;
%net = dagnn.DagNN.loadobj(netStruct.net) ;
%net.layers(176).block.loss = 'labelsmooth';
%net.layers(174).block.rate = 0.75;
net.conserveMemory = true;
net.meta.normalization.averageImage = reshape([105.6920,99.1345,97.9152],1,1,3);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 16;
opts.train.continue = false;
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = './data/res52_drop0.75_batch16_baseline5';
opts.train.derOutputs = {'objective', 1} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,20),0.01*ones(1,5)] ;
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;
% set for gan
clf;
global re;
global fa;
global re_num;
global gan_num;
global iter_re;
global ganPath;
ganPath = fullfile(opts.expDir, 'net-lsro.pdf') ;
re = 0;
fa = 0;
re_num = 0;
gan_num = 0;
iter_re = 0;
% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
% --------------------------------------------------------------------
im_url = imdb.images.data(batch) ;
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.85,1],...
    'Interpolation', 'bicubic','NumThreads',8);
labels = imdb.images.label(batch);
oim = bsxfun(@minus,im{1},opts.averageImage);
inputs = {'data',gpuArray(oim),'label',labels};
