import os,sys
import layer_tools as lt
import caffe
from caffe import params as P
from caffe import layers as L
import numpy as np

""" this utility is meant to build a U-Net using ResNet modules. """

def root_data_layer( net, batch_size, config, filler_name ):
    net.data, net.label = L.ROOTData( ntop=2, batch_size=batch_size, filler_config=config, filler_name=filler_name )
    return [net.data],net.label

def buildnet(nclasses=3,cfgfile="filler.cfg",batch_size=2,base_numout=16,nchannels=1,net_type="",use_batch_norm=True):

    net = caffe.NetSpec()
    if net_type not in ["train","test","deploy"]:
        raise ValueError("Net type must be one of "+str(net_type))
        return
    if net_type=="train":
        train = True
    else:
        train = False

    data_layers,label = root_data_layer( net, batch_size, cfgfile, net_type )

    # several layers of ResNet, with maxpooling occuring
    
    # Contracting ends

    # resnet group 1
    in1  = 1
    out1 = base_numout
    bn1   = np.maximum( out1/4, 4 )
    resnet1a  = lt.resnet_module( net, data_layers[0],  "resnet1a",  in1, 3, 1, 1,bn1,out1, True, train) # always batch_norm
    resnet1b  = lt.resnet_module( net,       resnet1a,  "resnet1b", out1, 3, 1, 1,bn1,out1, True, train) # always batch_norm
    maxpool1  = lt.pool_layer(    net,       resnet1b,  "maxpool1",    3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )
    
    # resnet group 2
    in2  = out1
    out2 = in2*2
    bn2  = np.maximum( out2/4, 4 )
    resnet2a  = lt.resnet_module( net,       maxpool1,  "resnet2a",  in2, 3, 1, 1,bn2,out2, use_batch_norm, train)
    resnet2b  = lt.resnet_module( net,       resnet2a,  "resnet2b", out2, 3, 1, 1,bn2,out2, use_batch_norm, train)
    maxpool2  = lt.pool_layer(    net,       resnet2b,  "maxpool2",    3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )

    # resnet group 3
    in3  = out2
    out3 = in3*2
    bn3  = np.maximum( out3/4, 4 )
    resnet3a  = lt.resnet_module( net,       maxpool2,  "resnet3a",  in3, 3, 1, 1,bn3,out3, use_batch_norm, train)
    resnet3b  = lt.resnet_module( net,       resnet3a,  "resnet3b", out3, 3, 1, 1,bn3,out3, use_batch_norm, train)
    maxpool3  = lt.pool_layer(    net,       resnet3b,  "maxpool3",    3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )

    # resnet group 4
    in4  = out3
    out4 = in4*2
    bn4  = np.maximum( out4/4, 4 )
    resnet4a  = lt.resnet_module( net,       maxpool3,  "resnet4a",  in4, 3, 1, 1,bn4,out4, use_batch_norm, train)
    resnet4b  = lt.resnet_module( net,       resnet4a,  "resnet4b", out4, 3, 1, 1,bn4,out4, use_batch_norm, train)
    maxpool4  = lt.pool_layer(    net,       resnet4b,  "maxpool4",    3, 2, P.Pooling.MAX, pad_w=0, pad_h=0 )

    # resnet group 5
    in5  = out4
    out5 = in5*2
    bn5  = np.maximum( out5/4, 4 )
    resnet5a  = lt.resnet_module( net,       maxpool4,  "resnet5a",  in5, 3, 1, 1,bn5,out5, use_batch_norm, train)
    resnet5b  = lt.resnet_module( net,       resnet5a,  "resnet5b", out5, 3, 1, 1,bn5,out5, use_batch_norm, train)

    # Expansive Part

    # resnet group 6
    unpool4   = lt.deconvolution_layer( net, resnet5b,  "unpool4", 4, 2, 1, out5, b_lr=0.0 )
    input6    = [ resnet4b, unpool4 ]  # skip connection
    merge5to6 = lt.concat_layer( net,      "concat56",     *input6 )
    resnet6a  = lt.resnet_module( net,      merge5to6,  "resnet6a",out4+out5, 3, 1, 1,bn4,out4, use_batch_norm, train)
    resnet6b  = lt.resnet_module( net,       resnet6a,  "resnet6b",     out4, 3, 1, 1,bn4,out4, use_batch_norm, train)

    # resnet group 7
    unpool3   = lt.deconvolution_layer( net, resnet6b,  "unpool3", 4, 2, 1, out4, b_lr=0.0 )
    input7    = [ resnet3b, unpool3 ]
    merge3to7 = lt.concat_layer(  net,     "concat37",     *input7 )
    resnet7a  = lt.resnet_module( net,      merge3to7,  "resnet7a", out3+out4, 3, 1, 1,bn3,out3, use_batch_norm, train)
    resnet7b  = lt.resnet_module( net,       resnet7a,  "resnet7b",      out3, 3, 1, 1,bn3,out3, use_batch_norm, train)

    # resnet group 8
    unpool2   = lt.deconvolution_layer( net, resnet7b,  "unpool2", 4, 2, 1, out3, b_lr=0.0 )
    input8    = [ resnet2b, unpool2 ]
    merge2to8 = lt.concat_layer(  net,     "concat28",     *input8 )
    resnet8a  = lt.resnet_module( net,      merge2to8,  "resnet8a", out2+out3, 3, 1, 1,bn2,out2, use_batch_norm, train)
    resnet8b  = lt.resnet_module( net,       resnet8a,  "resnet8b",      out2, 3, 1, 1,bn2,out2, use_batch_norm, train)

    # resnet group 9
    unpool1   = lt.deconvolution_layer( net, resnet8b,  "unpool1", 4, 2, 1, out2, b_lr=0.0 )
    input9    = [ resnet1b, unpool1 ]
    merge1to9 = lt.concat_layer(  net,     "concat19",     *input9 )
    resnet9a  = lt.resnet_module( net,      merge1to9,  "resnet9a", out1+out2, 3, 1, 1,bn1,out1, use_batch_norm, train)
    resnet9b  = lt.resnet_module( net,       resnet9a,  "resnet9b",      out1, 3, 1, 1,bn1,out1, use_batch_norm, train)

    # 1 x 1
    score     = lt.convolution_layer( net, resnet9b, "score", "score", nclasses+1, 1, 1, 0, 0.0, 
                                      addbatchnorm=False, train=train, add_relu=False, w_lr=1.0, b_lr=2.0, w_decay_mult=1.0, b_decay_mult=0.0 )

    # crop
    offset = 0
    crop_score = L.Crop( score, label, crop_param=dict(axis=2,offset=offset) )
    net.__setattr__("crop_score", crop_score)

    # softmax
    if net_type in ["train","test"]:
        softmaxloss   = L.SoftmaxWithLoss( crop_score, net.label, loss_param=dict(normalize=True, class_loss_weights=[1,1000,1000,1000]) )
        net.__setattr__( "softmaxloss", softmaxloss )

        acc       = L.Accuracy( crop_score, net.label, accuracy_param=dict(top_k=1,ignore_label=0) )
        net.__setattr__("accuracy", acc )

    elif net_type in ["deploy"]:
        softmax   = L.Softmax( crop_score )
        net.__setattr__("softmax",softmax)


    # then deconv layers + skip connections
    
    return net

if __name__=="__main__":

    # build network
    test_net  = buildnet(net_type="test")
    train_net = buildnet(net_type="train")
    deploy_net = buildnet(net_type="deploy")

    testout   = open('ub_uresnet_test.prototxt','w')
    print >> testout, "name: \"uB-U-ResNet\""
    print >> testout, test_net.to_proto()
    testout.close()

    trainout   = open('ub_uresnet_train.prototxt','w')
    print >> trainout, "name: \"uB-U-ResNet\""
    print >> trainout, train_net.to_proto()
    trainout.close()

    deployout   = open('ub_uresnet_deploy.prototxt','w')
    print >> deployout, "name: \"uB-U-ResNet\""
    print >> deployout, deploy_net.to_proto()
    deployout.close()




