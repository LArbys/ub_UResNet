import os,sys
import layer_tools as lt
import caffe
from caffe import params as P
from caffe import layers as L

""" this utility is meant to build a U-Net using ResNet modules. """

def root_data_layer( net, batch_size, config, filler_name ):
    net.data, net.label = L.ROOTData( ntop=2, batch_size=batch_size, filler_config=config, filler_name=filler_name )
    return [net.data],net.label

def buildnet(nclasses=3,cfgfile="filler.cfg",batch_size=1,base_numout=64,nchannels=1,train=True,use_batch_norm=False):

    net = caffe.NetSpec()
    net_type = "test"
    if train:
        net_type = "train"

    data_layers,label = root_data_layer( net, batch_size, cfgfile, net_type )

    # several layers of ResNet, with maxpooling occuring
    
    # Contracting ends

    # resnet group 1
    resnet1a  = lt.resnet_module( net, data_layers[0],  "resnet1a", 1,  3, 1, 1,8,64, use_batch_norm, train)
    resnet1b  = lt.resnet_module( net,       resnet1a,  "resnet1b", 64, 3, 1, 1,8,64, use_batch_norm, train)
    maxpool1  = lt.pool_layer(    net,       resnet1b,  "maxpool1",  3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )
    
    # resnet group 2
    resnet2a  = lt.resnet_module( net,       maxpool1,  "resnet2a", 64, 3, 1, 1,16,128, use_batch_norm, train)
    resnet2b  = lt.resnet_module( net,       resnet2a,  "resnet2b",128, 3, 1, 1,16,128, use_batch_norm, train)
    maxpool2  = lt.pool_layer(    net,       resnet2b,  "maxpool2",  3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )

    # resnet group 3
    resnet3a  = lt.resnet_module( net,       maxpool2,  "resnet3a",128, 3, 1, 1,32,256, use_batch_norm, train)
    resnet3b  = lt.resnet_module( net,       resnet3a,  "resnet3b",256, 3, 1, 1,32,256, use_batch_norm, train)
    maxpool3  = lt.pool_layer(    net,       resnet3b,  "maxpool3",  3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )

    # resnet group 4
    resnet4a  = lt.resnet_module( net,       maxpool3,  "resnet4a",256, 3, 1, 1,64,512, use_batch_norm, train)
    resnet4b  = lt.resnet_module( net,       resnet4a,  "resnet4b",512, 3, 1, 1,64,512, use_batch_norm, train)
    maxpool4  = lt.pool_layer(    net,       resnet4b,  "maxpool4",  3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )

    # resnet group 5
    resnet5a  = lt.resnet_module( net,       maxpool4,  "resnet5a",512, 3, 1, 1,128,1024, use_batch_norm, train)
    resnet5b  = lt.resnet_module( net,       resnet5a,  "resnet5b",1024,3, 1, 1,128,1024, use_batch_norm, train)

    # Expansive Part

    # resnet group 6
    unpool4   = lt.deconvolution_layer( net, resnet5b,  "unpool4", 2, 2, 0, 1024 )
    input6    = [ resnet4b, unpool4 ]  # skip connection
    merge5to6 = lt.concat_layer( net,      "concat56",     *input6 )
    resnet6a  = lt.resnet_module( net,      merge5to6,  "resnet6a",512+1024,3, 1, 1,64,512, use_batch_norm, train)
    resnet6b  = lt.resnet_module( net,       resnet6a,  "resnet6b",     512,3, 1, 1,64,512, use_batch_norm, train)

    # resnet group 7
    unpool3   = lt.deconvolution_layer( net, resnet6b,  "unpool3", 2, 2, 0, 512 )
    input7    = [ resnet3b, unpool3 ]
    merge3to7 = lt.concat_layer(  net,     "concat37",     *input7 )
    resnet7a  = lt.resnet_module( net,      merge3to7,  "resnet7a", 256+512,3, 1, 1,32,256, use_batch_norm, train)
    resnet7b  = lt.resnet_module( net,       resnet7a,  "resnet7b",     256,3, 1, 1,32,256, use_batch_norm, train)

    # resnet group 8
    unpool2   = lt.deconvolution_layer( net, resnet7b,  "unpool2", 2, 2, 0, 128 )
    input8    = [ resnet2b, unpool2 ]
    merge2to8 = lt.concat_layer(  net,     "concat28",     *input8 )
    resnet8a  = lt.resnet_module( net,      merge2to8,  "resnet8a", 128+256,3, 1, 1,16,128, use_batch_norm, train)
    resnet8b  = lt.resnet_module( net,       resnet8a,  "resnet8b",     128,3, 1, 1,16,128, use_batch_norm, train)

    # resnet group 9
    unpool1   = lt.deconvolution_layer( net, resnet8b,  "unpool1", 2, 2, 0, 64 )
    input9    = [ resnet1b, unpool1 ]
    merge1to9 = lt.concat_layer(  net,     "concat19",     *input9 )
    resnet9a  = lt.resnet_module( net,      merge1to9,  "resnet9a",  64+128,3, 1, 1, 8,64, use_batch_norm, train)
    resnet9b  = lt.resnet_module( net,       resnet9a,  "resnet9b",     128,3, 1, 1, 8,64, use_batch_norm, train)

    # 1 x 1
    score     = lt.convolution_layer( net, resnet9b, "score", "score", nclasses+1, 1, 1, 0, 0.0, addbatchnorm=False, train=train )

    # crop
    offset = 0
    crop_score = L.Crop( score, label, crop_param=dict(axis=2,offset=offset) )
    net.__setattr__("crop_score", crop_score)

    # softmax
    #if train:
    softmaxloss   = L.SoftmaxWithLoss( crop_score, net.label, loss_param=dict(normalize=True, class_loss_weights=[1,10000,10000,10000]) )
    net.__setattr__( "softmaxloss", softmaxloss )

    acc       = L.Accuracy( crop_score, net.label )
    net.__setattr__("accuracy", acc )
    

    # then deconv layers + skip connections
    
    return net

if __name__=="__main__":

    # build network
    test_net  = buildnet(train=False)
    train_net = buildnet(train=True)

    testout   = open('ub_uresnet_test.prototxt','w')
    print >> testout, "name: \"uB-U-ResNet\""
    print >> testout, test_net.to_proto()
    testout.close()

    trainout   = open('ub_uresnet_train.prototxt','w')
    print >> trainout, "name: \"uB-U-ResNet\""
    print >> trainout, train_net.to_proto()
    trainout.close()



