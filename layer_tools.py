import os,sys
from caffe import layers as L
from caffe import params as P

def convolution_layer( net, input_layer, layername_stem, parname_stem, noutputs, stride, kernel_size, pad, init_bias, 
                       addbatchnorm=True, train=True, kernel_w=None, kernel_h=None, pad_w=None, pad_h=None, 
                       w_decay_mult=1.0, b_decay_mult=1.0,
                       w_lr=1.0, b_lr=1.0, add_relu=True ):
    if kernel_w is None or kernel_h is None:
        if pad_w is None:
            my_pad_w = pad
        else:
            my_pad_w = pad_w
        if pad_h is None:
            my_pad_h = pad
        else:
            my_pad_h = pad_h
        
        # square convolution
        conv = L.Convolution( input_layer, 
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_h=my_pad_h,
                              pad_w=my_pad_w,
                              num_output=noutputs,
                              weight_filler=dict(type="msra"),
                              bias_filler=dict(type="constant",value=init_bias),
                              param=[dict(name="par_%s_conv_w"%(parname_stem),lr_mult=w_lr,decay_mult=w_decay_mult),
                                     dict(name="par_%s_conv_b"%(parname_stem),lr_mult=b_lr,decay_mult=b_decay_mult)] )
    else:
        conv = L.Convolution( input_layer, 
                              kernel_w=kernel_w,
                              kernel_h=kernel_h,
                              stride=stride,
                              pad_h=pad_h,
                              pad_w=pad_w,
                              num_output=noutputs,
                              weight_filler=dict(type="msra"),
                              bias_filler=dict(type="constant",value=init_bias),
                              param=[dict(name="par_%s_conv_w"%(parname_stem),lr_mult=w_lr,decay_mult=w_decay_mult),
                                     dict(name="par_%s_conv_b"%(parname_stem),lr_mult=b_lr,decay_mult=b_decay_mult)] )
        
    net.__setattr__( layername_stem+"_conv", conv )
    if addbatchnorm:
        if train:
            conv_bn = L.BatchNorm( conv, in_place=True, batch_norm_param=dict(use_global_stats=False),param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            conv_bn = L.BatchNorm( conv,in_place=True,batch_norm_param=dict(use_global_stats=True))
        conv_scale = L.Scale( conv_bn, in_place=True, scale_param=dict(bias_term=True))
        conv_relu  = L.ReLU(conv_scale,in_place=True)
        net.__setattr__( layername_stem+"_bn", conv_bn )
        net.__setattr__( layername_stem+"_scale", conv_scale )
        net.__setattr__( layername_stem+"_relu", conv_relu )
        nxtlayer = conv_relu
    else:
        if add_relu:
            conv_relu  = L.ReLU( conv, in_place=True )
            net.__setattr__( layername_stem+"_relu", conv_relu )
            nxtlayer = conv_relu
        else:
            nxtlayer = conv
    return nxtlayer

def concat_layer( net, layername, *bots ):
    convat = L.Concat(*bots, concat_param=dict(axis=1))
    net.__setattr__( "%s_concat"%(layername), convat )
    return convat
            
def final_fully_connect( net, bot, name, nclasses=2, lr_mult=1.0 ):
    fc2 = L.InnerProduct( bot, num_output=nclasses, weight_filler=dict(type='msra'),param=dict(lr_mult=lr_mult))
    net.__setattr__( name, fc2 )
    return fc2

def resnet_module( net, bot, name, ninput, kernel_size, stride, pad, bottleneck_nout, expand_nout, use_batch_norm, train ):
    if ninput!=expand_nout:
        bypass_conv = L.Convolution( bot,
                                     kernel_size=1,
                                     stride=1,
                                     num_output=expand_nout,
                                     pad=0,
                                     bias_term=False,
                                     weight_filler=dict(type="msra") )
        if use_batch_norm:
            if train:
                bypass_bn = L.BatchNorm(bypass_conv,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                        param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
            else:
                bypass_bn = L.BatchNorm(bypass_conv,in_place=True,batch_norm_param=dict(use_global_stats=True))
            bypass_scale = L.Scale(bypass_bn,in_place=True,scale_param=dict(bias_term=True))
            net.__setattr__(name+"_bypass",bypass_conv)
            net.__setattr__(name+"_bypass_bn",bypass_bn)
            net.__setattr__(name+"_bypass_scale",bypass_scale)
        else:
            net.__setattr__(name+"_bypass",bypass_conv)
        bypass_layer = bypass_conv
    else:
        bypass_layer  = bot

    # bottle neck
    bottleneck_layer = L.Convolution(bot,num_output=bottleneck_nout,kernel_size=1,stride=1,pad=0,bias_term=False,weight_filler=dict(type="msra"))
    if use_batch_norm:
        if train:
            bottleneck_bn    = L.BatchNorm(bottleneck_layer,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                           param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            bottleneck_bn    = L.BatchNorm(bottleneck_layer,in_place=True,batch_norm_param=dict(use_global_stats=True))
        bottleneck_scale = L.Scale(bottleneck_bn,in_place=True,scale_param=dict(bias_term=True))
        bottleneck_relu  = L.ReLU(bottleneck_scale,in_place=True)
    else:
        bottleneck_relu  = L.ReLU(bottleneck_layer,in_place=True)
    net.__setattr__(name+"_btlnk",bottleneck_layer)
    if use_batch_norm:
        net.__setattr__(name+"_btlnk_bn",bottleneck_bn)
        net.__setattr__(name+"_btlnk_scale",bottleneck_scale)
    net.__setattr__(name+"_btlnk_relu",bottleneck_relu)

    # conv
    conv_layer = L.Convolution(bottleneck_relu,num_output=bottleneck_nout,kernel_size=3,stride=1,pad=1,bias_term=False,weight_filler=dict(type="msra"))
    if use_batch_norm:
        if train:
            conv_bn    = L.BatchNorm(conv_layer,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                     param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            conv_bn    = L.BatchNorm(conv_layer,in_place=True,batch_norm_param=dict(use_global_stats=True))
        conv_scale = L.Scale(conv_bn,in_place=True,scale_param=dict(bias_term=True))
        conv_relu  = L.ReLU(conv_scale,in_place=True)
    else:
        conv_relu  = L.ReLU(conv_layer,in_place=True)
    net.__setattr__(name+"_conv",conv_layer)
    if use_batch_norm:
        net.__setattr__(name+"_conv_bn",conv_bn)
        net.__setattr__(name+"_conv_scale",conv_scale)
    net.__setattr__(name+"_conv_relu",conv_relu)

    # expand
    expand_layer = L.Convolution(conv_relu,num_output=expand_nout,kernel_size=1,stride=1,pad=0,bias_term=False,weight_filler=dict(type="msra"))
    ex_last_layer = expand_layer
    if use_batch_norm:
        if train:
            expand_bn    = L.BatchNorm(expand_layer,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                       param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
        else:
            expand_bn    = L.BatchNorm(expand_layer,in_place=True,batch_norm_param=dict(use_global_stats=True))
        expand_scale = L.Scale(expand_bn,in_place=True,scale_param=dict(bias_term=True))
        ex_last_layer = expand_scale
    net.__setattr__(name+"_expnd",expand_layer)
    if use_batch_norm:
        net.__setattr__(name+"_expnd_bn",expand_bn)
        net.__setattr__(name+"_expnd_scale",expand_scale)

    # Eltwise
    elt_layer = L.Eltwise(bypass_layer,ex_last_layer, eltwise_param=dict(operation=P.Eltwise.SUM))
    elt_relu  = L.ReLU( elt_layer,in_place=True)
    net.__setattr__(name+"_eltwise",elt_layer)
    net.__setattr__(name+"_eltwise_relu",elt_relu)
    

    return elt_relu
                                      
def data_layer_stacked( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=-1 ):
    transform_pars = {"mean_file":mean_file,
                      "mirror":False}
    if crop_size>0:
        transform_pars["crop_size"] = crop_size
    if net_type in ["train","test"]:
        net.data, net.label = L.Data(ntop=2,backend=P.Data.LMDB,source=inputdb,batch_size=batch_size,transform_param=transform_pars)
    elif net_type=="deploy":
        #net.data, net.label = L.MemoryData(ntop=2,batch_size=batch_size, height = height, width = width, channels = nchannels)
        pydata_params = dict(configfile="config.yaml")
        pylayer = 'UBHiResData'
        net.data, net.label,net.eventid = L.Python(module='layers.ubhiresdata', layer=pylayer, ntop=3, param_str=str(pydata_params))
    return [net.data], net.label

def data_layer_trimese( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, slice_points, crop_size=-1 ):
    data, label = data_layer_stacked( net, inputdb, mean_file, batch_size, net_type, height, width, nchannels, crop_size=crop_size )
    slices = L.Slice(data[0], ntop=3, name="data_trimese", slice_param=dict(axis=1, slice_point=slice_points))
    #for n,slice in enumerate(slices):
    #    net.__setattr__( slice, "data_plane%d"%(n) )

    return slices, label
    
def pool_layer( net, inputlayer, layername, kernel_size, stride, pooltype=P.Pooling.MAX, pad_w=0, pad_h=0 ):
    pooll = L.Pooling(inputlayer, kernel_size=kernel_size, stride=stride, pool=pooltype, pad_w=pad_w, pad_h=pad_h)
    net.__setattr__( layername, pooll )
    return pooll

def deconvolution_layer( net, inputlayer, layername, kernel_size, stride, pad, num_output, w_lr=1.0, b_lr=1.0, init_bias=0.0 ):
    parname_stem = layername
    deconv = L.Deconvolution( inputlayer, 
                              convolution_param=dict(num_output=num_output, group=num_output, 
                                                     kernel_size=kernel_size, 
                                                     stride=stride, pad=pad,
                                                     weight_filler=dict(type="bilinear"),
                                                     bias_filler=dict(type="constant",value=init_bias)),
                              param=[dict(name="par_%s_deconv_w"%(parname_stem),lr_mult=w_lr),dict(name="par_%s_conv_b"%(parname_stem),lr_mult=b_lr)] )
    net.__setattr__( layername, deconv )
    return deconv
