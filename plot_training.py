import os,sys
sys.argv.append("-b")
import ROOT as rt
import numpy as np

ntraining =  9471
nevents_per_iteration = 100
avesize = 20

iters_per_point = 1
test_iters_per_point = 10
nevents_per_epoch = float(ntraining)/float(nevents_per_iteration)

f = open(sys.argv[1],'r')

out = rt.TFile("plots_training.root", "recreate")

train_loss_pts = []
train_acc_pts = []
test_loss_pts = []
test_acc_pts = []

niter = 0
niter_test = 0
niter_trainacc = 0
niter_testloss = 0
loss_max = 0
for l in f:
    if "solver.cpp:244" in l and "Train net output #1: softmaxloss" in l:
        # TRAINING LOSS
        loss = float( l.strip().split("=")[1].split("(")[0].strip() )
        train_loss_pts.append( (niter,loss) )
        niter += iters_per_point
        if loss_max<loss:
            loss_max = loss
    elif "Test net output #0: accuracy" in l:
        # VAL ACC
        try:
            acc = float( l.strip().split("=")[1] )
        except:
            acc = 0.0
        if acc!=acc:
            if len(test_acc_pts)>0:
                acc = test_acc_pts[-1][1]
            else:
                acc = 0.0
        test_acc_pts.append( (niter_test,acc) )
        niter_test += test_iters_per_point
    elif "Train net output #0: accuracy" in l:
        # TRAIN ACC
        train_acc = float( l.strip().split("=")[1] )
        if train_acc!=train_acc:
            if len(train_acc_pts)>0:
                train_acc = train_acc_pts[-1][1]
            else:
                train_acc = 0.0
        train_acc_pts.append( (niter_trainacc, train_acc ) )
        niter_trainacc += iters_per_point
    elif "Test net output #1: softmaxloss" in l:
        test_loss = float(l.strip().split("=")[1].split()[0])
        if test_loss!=test_loss:
            if len(test_loss_pts)>0:
                test_loss = test_loss_pts[-1][1]
            else:
                test_loss = 0.0
        test_loss_pts.append( (niter_testloss, test_loss) )
        niter_testloss += test_iters_per_point

# make ave. points
ave_pts = {}
for name,data in [ ("train_acc_ave",train_acc_pts),("train_loss_ave",train_loss_pts) ]:
    npts = 0
    ave = 0.0
    running_ave = []
    for pt in data:
        npts += 1
        if npts<avesize:
            ave = ave*(npts-1) + pt[1]
            ave /= npts
        else:
            ave = ave*(avesize-1) + pt[1]
            ave /= avesize
        running_ave.append(ave)
    ave_pts[name] = running_ave
    

gloss = rt.TGraph( len(train_loss_pts) )
gloss_ave = rt.TGraph( len(train_loss_pts) )
for n,pt in enumerate(train_loss_pts):
    gloss.SetPoint(n, float(pt[0])/nevents_per_epoch, pt[1])
    gloss_ave.SetPoint(n, float(pt[0])/nevents_per_epoch, ave_pts["train_loss_ave"][n])

gacc = rt.TGraph( len(test_acc_pts) )
for n,pt in enumerate(test_acc_pts):
    gacc.SetPoint(n,float(pt[0])/nevents_per_epoch, pt[1])

gacc_train = rt.TGraph( len(train_acc_pts) )
gacc_train_ave = rt.TGraph( len(train_acc_pts) )
for n,pt in enumerate(train_acc_pts):
    gacc_train.SetPoint(n,float(pt[0])/nevents_per_epoch, pt[1])
    gacc_train_ave.SetPoint(n,float(pt[0])/nevents_per_epoch, ave_pts["train_acc_ave"][n])

gloss_test = rt.TGraph(len(test_loss_pts))
for n,pt in enumerate(test_loss_pts):
    gloss_test.SetPoint(n,float(pt[0])/nevents_per_epoch, pt[1] )

gloss.Write("gloss")
gacc.Write("gacc_test")
gacc_train.Write("gacc_train")
gloss_test.Write("gloss_test")

c = rt.TCanvas("c","c",1200,600)
c.Divide(2,1)
c.cd(1).SetLogy(1)
c.cd(1).SetGridx(1)
c.cd(1).SetGridy(1)
gloss.SetMarkerStyle(24)
gloss.SetMarkerSize(0.5)
gloss.SetMarkerColor(rt.kBlue-9)
gloss.Draw("AP")
gloss.SetTitle("Loss Curve;epochs;loss")
gloss.GetYaxis().SetRangeUser(0.1e-2,1.1*loss_max)
gloss_test.SetMarkerStyle(25)
gloss_test.SetMarkerColor(rt.kRed)
gloss_test.SetMarkerSize(0.5)
gloss_test.SetLineColor(rt.kRed)
gloss_test.Draw("P")

# (ave)
gloss_ave.SetLineWidth(2)
gloss_ave.SetLineColor(rt.kBlue)
gloss_ave.Draw("L")

# (legend)

loss_legend = rt.TLegend(0.15, 0.15, 0.6, 0.45)
loss_legend.SetBorderSize(1)
loss_legend.AddEntry( gloss,      "Train  loss", "P" )
loss_legend.AddEntry( gloss_ave,  "Train  loss (ave.)", "L" )
loss_legend.AddEntry( gloss_test, "Test loss", "P")
loss_legend.Draw()

# ACCURACY
c.cd(2)
c.cd(2).SetGridx(1)
c.cd(2).SetGridy(1)
gacc_train.Draw("AP")
gacc_train.SetMarkerStyle(24)
gacc_train.SetMarkerSize(0.5)
gacc_train.SetMarkerColor(rt.kBlue-9)
gacc_train.GetYaxis().SetRangeUser(0,1)
gacc_train.SetTitle("Accuracy;epochs;accuracy")

gacc.SetMarkerStyle(25)
gacc.SetMarkerSize(0.5)
gacc.SetMarkerColor(rt.kRed)
gacc.SetLineColor(rt.kRed)
gacc.SetLineWidth(2)
gacc.Draw("P")

# (averages)
gacc_train_ave.SetLineWidth(1)
gacc_train_ave.SetLineColor(rt.kBlue)
gacc_train_ave.Draw("LP")

# (legend)
acc_legend = rt.TLegend(0.45, 0.15, 0.85, 0.45)
acc_legend.SetBorderSize(1)
acc_legend.AddEntry( gacc_train, "Train  accuracy", "P" )
acc_legend.AddEntry( gacc_train_ave,  "Train  accuracy (ave.)", "L" )
acc_legend.AddEntry( gacc, "Test accuracy", "P")
acc_legend.Draw()

c.SaveAs("training_plot.png")

print "number of points: ",len(train_loss_pts)," nepochs=",float(niter)/nevents_per_epoch
