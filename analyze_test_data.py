import caffe
import numpy as np
import ROOT as rt
from math import log
from array import array
from larcv import larcv
from countpe import countpe
import time
import sys
import cv2

# This script setups caffe in TEST mode

gpu_id = 0
caffe.set_device(gpu_id)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

prototxt = "ub_uresnet_deploy.prototxt"
model = "snapshot_rmsprop_uburn_iter_4100.caffemodel"
#model = "saved_states/smallsampletest_snapshot_rmsprop_uburn_iter_3000.caffemodel"
tpcproducer = "tpc_hires_crop"
pmtproducer = "pmt"
out_tag = "test"
save_images = True

# v0 data files
# REMEMBER: this file list must match the one in filler_test.cfg -- we are trying to get model scores and also calcualte PMT values that are of the correct entry
rootfiles = ["/mnt/raid0/taritree/v0/hires_test/bnb_mc_hirescrop_valid.root"]
#rootfiles = ["/mnt/raid0/taritree/v0/hires_test/bnb_mc_hirescrop_train.root"]
#rootfiles = ["ex_data/larcv_hirescrop_out_0010.root"]

net = caffe.Net( prototxt, model, caffe.TEST )
input_shape = net.blobs["data"].data.shape
batch_size = input_shape[0]

class_name  = ["bg","shower","MIP","HIP"]
color_index = [-1,1,0,2] # BGR format
color_dict  = {"bg":-1,
               "shower":1,
               "MIP":0,
               "HIP":2}

#nevents = 36
nevents = 100

# setup input files string for PSet
str_input = "["
for r in rootfiles:
    str_input += r
    if r != rootfiles[-1]:
        str_input+=","
str_input += "]"
print str_input

# ROOT data
from ROOT import std
str_parname  = std.string( "IOMan2" )
#iocfg = larcv.PSet(str_parname,str_iomancfg)
iocfg = larcv.PSet("IOMan2")
iocfg.add_value( "Name", "IOMan2" )
iocfg.add_value( "IOMode", "0" )
iocfg.add_value( "Verbosity", "2" )
iocfg.add_value( "InputFiles", str_input )
iocfg.add_value( "ReadOnlyType", "[0,0,0,1]" )
iocfg.add_value( "ReadOnlyName", "[tpc_hires_crop,segment_hires_crop,pmt,tpc_hires_crop]" )

ioman = larcv.IOManager( iocfg )
ioman.initialize()
if nevents is None:
    nevents = ioman.get_n_entries()

print "Network Ready: Batch Size=",batch_size
print "[ENTER] to continue."
raw_input()


# setup output

out = rt.TFile("out_%s_netanalysis.root"%(out_tag), "RECREATE" )
# file entry number
entry = array('i',[0])
# event ID
run = array('i',[0])
subrun = array('i',[0])
eventid = array('i',[0])
# truth information
mode = array('i',[0])
current = array('i',[0])
enu  = array('f',[0])
vtx  = array('f',[0]*3)
pe   = array('f',[0.0])
peped = array('f',[0.])

# label statistics
npixels   = array('i',[0])           # number of interesting pixels
nlabels   = array('i',[0,0,0,0])     # bg, shower, mip, proton
npredict  = array('i',[0,0,0,0])
label_acc = array('f',[0.,0.,0.,0.])
label_q   = array('f',[0.,0.,0.,0.])
predict_q = array('f',[0.,0.,0.,0.])
total_acc = array('f',[0.])

tree = rt.TTree("net","net output")
tree.Branch("entry",entry,"entry/I")
tree.Branch("run", run, "run/I")
tree.Branch("subrun", subrun, "subrun/I")
tree.Branch("event", eventid, "event/I")
tree.Branch("current", current, "current/I")
tree.Branch("mode", mode, "mode/I")
tree.Branch("enu", enu, "enu/F")
tree.Branch("vtx", vtx, "vtx[3]/F")
tree.Branch("pe", pe, "pe/F" )
tree.Branch("peped", peped, "peped/F" )
tree.Branch("npixels",npixels,"npixels/I")
tree.Branch("nlabels",nlabels,"nlabels[4]/I")
tree.Branch("npredict",npredict,"npredict[4]/I")
tree.Branch("total_acc",total_acc,"total_acc/F")
tree.Branch("label_acc",label_acc,"label_acc[4]/F")
tree.Branch("label_q",label_q,"label_q[4]/F")
tree.Branch("predict_q",predict_q,"predict_q[4]/F")


nbatches = nevents/batch_size
if nevents%batch_size!=0:
    nbatches += 1
filler = larcv.ThreadFillerFactory.get_filler("deploy")

ibatch = 0
ievents = 0

while ibatch<nbatches:
    print "batch ",ibatch," of ",nbatches
    keys = []

    # pass through network
    net.forward() 
    data   = net.blobs["data"].data
    labels = net.blobs["label"].data
    score_raw = net.blobs["crop_score"].data
    scores = net.blobs["softmax"].data
    processed = filler.processed_entries()
    print "number of process entries: ",processed.size()
    print " data: ",data.shape
    print " labels: ",labels.shape
    print " scores: ",scores.shape

    # evaluate each image
    for b in range(batch_size):
        print "Image ",(ibatch,b,ievents)
        file_entry = processed[b]
        ioman.read_entry( file_entry  )

        # LArCV info
        eventroi = ioman.get_data( larcv.kProductROI, "tpc_hires_crop" )
        current[0] = eventroi.at(0).NuCurrentType()
        mode[0]    = eventroi.at(0).NuInteractionType()
        entry[0]   = ievents
        run[0]     = eventroi.run()
        subrun[0]  = eventroi.subrun()
        eventid[0] = eventroi.event()
        pe[0],maxpmtch, peped[0] = countpe( ioman, pmtproducer )

        # look for neutrino ROI
        enu[0] = -1.0
        vtx[0] = 0.0
        vtx[1] = 0.0
        vtx[2] = 0.0
        for roi in eventroi.ROIArray():
            if roi.PdgCode()==14 or roi.PdgCode()==12:
                enu[0] = roi.EnergyInit()
                vtx[0] = roi.X()
                vtx[1] = roi.Y()
                vtx[2] = roi.Z()
                break

        for i in range(0,4):
            label_q[i] = 0.0
            predict_q[i] = 0.0

        #out_image = np.zeros( (2*data.shape[2]+1,2*data.shape[3]+1,3) )
        #out_image[data.shape[2],:,:] = 100.0
        #out_image[:,data.shape[3],:] = 100.0
        out_image = np.zeros( ( data.shape[2], 2*data.shape[3]+1, 3 ) )
        out_image[:,data.shape[2],:] = 100.0

        above_thresh = np.where( data[b,0,:,:]>0 )
        npixels[0] = len(above_thresh[0])
        particle = np.where( scores[b,0,:,:]<0.5 )
        print " nabove thresh=",npixels[0]," frac=",float(npixels[0])/(data.shape[2]*data.shape[3])
        print " not predicted background=",len(particle[0])
        if npixels[0]==0:
            npixels[0]
            total_acc[0] = -1.0
            for i in range(0,4):
                nlabels[i] = 0
                npredict[i] = 0
                label_acc[i] = -1.0
            tree.Fill()
            continue

        # make output image
        out_image[0:data.shape[2],0:data.shape[3],0]   = data[b,0,:,:]         # data image
        out_image[0:data.shape[2],0:data.shape[3],1]   = data[b,0,:,:]         # data image
        out_image[0:data.shape[2],0:data.shape[3],2]   = data[b,0,:,:]         # data image
        #out_image[data.shape[2]+1:,0:data.shape[3]]  = scores[b,1,:,:]*100.0 # MIP image
        #out_image[0:data.shape[2],data.shape[3]+1:]  = scores[b,2,:,:]*100.0 # Shower image
        #out_image[data.shape[2]+1:,data.shape[3]+1:] = scores[b,3,:,:]*100.0 # HIP image
        out_offset = {0:(0,0),1:(0,data.shape[2]+1),2:(data.shape[2]+1,0),3:(data.shape[2]+1,data.shape[3]+1)}
        offset = data.shape[3]+1

        # summarize
        all_ncorrect = 0
        class_correct = [0,0,0,0]
        class_npixels = [0,0,0,0]
        class_npredict = [0,0,0,0]
        class_nanswer  = [0,0,0,0]
        n_nonbg = 0

        for x,y in zip(above_thresh[0],above_thresh[1]):
            answer = (int)(labels[b,0,x,y]+0.1)
            predict = np.argmax( scores[b,:,x,y] )
            nbg_predict = np.argmax( scores[b,1:,x,y] )+1
            #predict = np.argmax( scores[b,1:,x,y] )+1
            #print "truth=",answer,"predict=",predict,scores[b,:,x,y]
            class_npixels[answer] += 1
            class_npredict[predict] += 1
            if answer==predict:
                class_correct[answer] += 1
                all_ncorrect += 1
            if predict!=0:
                n_nonbg += 1
            label_q[answer] += data[b,0,x,y]
            predict_q[predict] += data[b,0,x,y]

            # color in image
            if answer in [1,3]:
                for iclass in [1,2,3]:
                    if iclass==answer:
                        out_image[ x, y, color_index[answer] ] = data[b,0,x,y]
                    else:
                        out_image[ x, y, color_index[iclass] ] = 0
            elif answer==2:
                # color in cyan if sample class
                out_image[ x, y, 0 ] = data[b,0,x,y]
                out_image[ x, y, 1 ] = data[b,0,x,y]
                out_image[ x, y, 2 ] = 0
                    
            # color in prediction
            #if predict in [1,3]:
            #    out_image[out_offset[predict][0]+x,out_offset[predict][1]+y,color_index[predict]] = scores[b,predict,x,y]*100.0
            #else:
            #    out_image[out_offset[predict][0]+x,out_offset[predict][1]+y,0] = scores[b,predict,x,y]*100.0
            #    out_image[out_offset[predict][0]+x,out_offset[predict][1]+y,1] = scores[b,predict,x,y]*100.0
            if predict in [1,3]:
                out_image[x,offset+y,color_index[predict]] = scores[b,predict,x,y]*100.0
            else:
                out_image[x,offset+y,0] = scores[b,predict,x,y]*100.0
                out_image[x,offset+y,1] = scores[b,predict,x,y]*100.0

        total_acc[0] = float(all_ncorrect)/float(npixels[0])
        print " correct (all classes): ",total_acc[0]
        print " non-bg predictions: ",n_nonbg

        for i,n in enumerate(class_npixels):
            nlabels[i] = n

        for iclass,ncorr in enumerate(class_correct):
            if class_npixels[iclass]>0:
                acc = float(ncorr)/float(class_npixels[iclass])
                print " correct[",class_name[iclass],"]: ",acc," of ",class_npixels[iclass]," pixels"
                label_acc[iclass] = acc
            else:
                print " correct[",class_name[iclass],"]: no true pixels."
                label_acc[iclass] = -1.0
        for iclass,npredicted in enumerate(class_npredict):
            print " predicted[",class_name[iclass],"]: ",float(npredicted)/float(npixels[0])," of ",class_npixels[iclass]," pixels"
            npredict[iclass] = npredicted

        if save_images:
            #if enu[0]<500 and current[0]==0:
            cv2.imwrite( "labelmaps/label_map_%03d_mode%04d.png"%(ievents,mode[0]), out_image )
        if ievents<nevents:
            tree.Fill()
        ievents += 1
        #raw_input()

    ibatch += 1

out.Write()
    
