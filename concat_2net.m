function net = concat_2net(net1,net2)
net1 = net1.saveobj();
net2 = net2.saveobj();
net.vars = [net1.vars,net2.vars]; 
net.layers = [net1.layers,net2.layers];
net.params = [net1.params,net2.params];
net.meta = net1.meta;
net = dagnn.DagNN.loadobj(net);
end