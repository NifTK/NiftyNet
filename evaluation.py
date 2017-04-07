import os
import os.path
import sys
import time
import scipy
import nibabel as nib
import numpy as np
import tensorflow as tf
from eval_dep import perform_evaluation
import itertools
import util


#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def run(param):
    list_names = None
    name_for_file = os.path.split(param.ref_image_dir)[1]+'_'+os.path.split(param.seg_image_dir)[1]
    print(name_for_file)
    if param.list_file == None and param.name_ref ==None or (len(param.name_ref)<1 and len(param.list_file)<1):
        namesSeg = util.list_nifti_files(param.seg_image_dir)
        namesRef = util.list_nifti_files(param.ref_image_dir)
    elif param.name_ref is not None :
        list_names = param.name_ref
    elif param.list_file is not None :
        text_file = open("param.list_file", "r")
        list_names = text_file.read().split(',')
    if list_names is not None:
        for name in list_names:
            namesRef_temp,namesSeg_temp = util.list_associations_nifti_files(param.ref_image_dir, param.seg_image_dir,name)
            if len(namesSeg_temp) != len(namesRef_temp):
                namesSeg_new = [item for item in namesSeg_temp for i in range(len(namesRef_temp))]
                namesRef_new = [namesRef_temp for i in range(len(namesSeg_temp))]
            else:
                namesSeg_new = namesSeg_temp
                namesRef_new = namesRef_temp
            namesSeg.append(namesSeg_new)
            namesRef.append(namesRef_new)
            name_for_file = name_for_file+'_'.join(str(e) for e in list_names)


    result_file = open(param.save_eval_dir + name_for_file+param.name_out +'.csv', 'w+')
    print("Names is " ,namesRef,namesSeg)
    print >> result_file ,"NameRef, NameSeg, Label, VolRef, VolSeg, TP, FP, FN, TPc, FPc, FNc, DSC, AvDist, VolDiff, OER, OEFP, OEFN, DE, DEFP, DEFN, PPV, NPV, FPR, Sens, Spec, Acc, Jaccard, HD"
    if len(namesRef)!=len(namesSeg):
        print('incompatibility in number of files, have to do all pairwise measurements')
        namesSeg_new = [item for item in namesSeg for i in range(len(namesRef))]
        namesRef_new = [namesRef for i in range(len(namesSeg))]
        namesSeg = namesSeg_new
        namesRef = list(itertools.chain.from_iterable(namesRef_new))

    print(namesSeg,namesRef)
    for i in range(0,len(namesSeg)):
        nameSeg = namesSeg[i]
        nameRef = namesRef[i]
        print (nameRef)
        SegNii = nib.load(os.path.join(param.seg_image_dir,nameSeg))
        RefNii = nib.load(os.path.join(param.ref_image_dir,nameRef))
        PixDim = SegNii.header.get_zooms()[0:3]
        Seg = SegNii.get_data()
        Ref = RefNii.get_data()
        if Seg.shape !=Ref.shape:
            print("Not possible to do comparison, have to go to next pair...")
            continue
        if param.type_eval == "binary" and np.max(Seg)<=1:
            Seg[Seg>param.threshold] = 1
            Seg[Seg<param.threshold] = 0
        uniqueRef = np.unique(Ref)
        uniqueSeg = np.unique(Seg)
        print(uniqueSeg,len(uniqueSeg))
        if len(uniqueSeg)==2: # Binary segmentation
            print('Binary analysis')
            typeAnalysis = 'Binary'
            PE = perform_evaluation(Seg,Ref,Seg,6,PixDim)
            TPc, FPc, FNc = PE.ConnectedElements()
            OER, OEFP, OEFN = PE.OE()
            DE, DEFP, DEFN = PE.DE()
            print >> result_file, '%s, %s, %d, %d, %d , %d, %d, %d, %d, %d, %d, %f, %f, %f, %f, %d, %d, %d, %d, %d, %f, %f, %f, %f, %f, %f, %f, %f'\
                         %(nameRef, nameSeg, 1, np.sum(Ref), np.sum(Seg), PE.TP(), PE.FP(), PE.FN(), TPc, FPc, FNc, PE.DSC(), PE.AvDist(),
                           PE.VolDiff(), OER, OEFP, OEFN, DE, DEFP, DEFN, PE.PPV(), PE.NPV(), PE.FPR(),  PE.Sensitivity(), PE.Specificity(), PE.Accuracy(), PE.Jaccard(), PE.HD()
                )
        elif len(uniqueRef)>2 and np.max(uniqueRef)>1: # Multiple labels
            typeAnalysis = 'Multiple labels'
            print ('Multiple labels analysis')
            for i in uniqueRef:
                if i==0:
                    continue
                SegB = np.copy(Seg)
                SegB[Seg!=i] = 0
                SegB[Seg==i] = 1
                RefB = np.copy(Ref)
                RefB[Ref!=i] = 0
                RefB[Ref==i] = 1
                if np.count_nonzero(SegB) == 0:
                    print("Nothing in element ")
                    continue
                PE = perform_evaluation(SegB, RefB, SegB, 6, PixDim)
                TPc, FPc, FNc = PE.ConnectedElements()
                OER, OEFP, OEFN = PE.OE()
                DE, DEFP, DEFN = PE.DE()
                HD = PE.HD()
                Jacc = PE.Jaccard()
                Acc = PE.Accuracy()
                Sens = PE.Sensitivity()
                Spec = PE.Specificity()
                print >> result_file, '%s, %s, %d, %d, %d , %d, %d, %d, %d, %d, %d, %f, %f, %f, %f, %d, %d, %d, %d, %d, %f, %f, %f, %f, %f, %f, %f, %f' \
                                      % (nameRef, nameSeg, 1, np.sum(Ref), np.sum(Seg), PE.TP(), PE.FP(), PE.FN(), TPc, FPc,
                                         FNc, PE.DSC(), PE.AvDist(),
                                         PE.VolDiff(), OER, OEFP, OEFN, DE, DEFP, DEFN, PE.PPV(), PE.NPV(), PE.FPR(),
                                         PE.Sensitivity(), PE.Specificity(), PE.Accuracy(), PE.Jaccard(), PE.HD()
                                         )
        else:
            typeAnalysis = 'Probability' # Probabilistic segmentation
            print('Probabilistic analysis by step of %f'%(float(param.step)))
            rangeCheck = np.arange(0,1,float(param.step))
            print(rangeCheck)
            for i in rangeCheck:
                SegB = np.copy(Seg)
                ones = np.ones_like(Seg)
                zeros = np.zeros_like(Seg)
                SegB = np.where(np.greater(Seg,i),ones,zeros)
                # SegB[Seg <= i] = 0
                # SegB[Seg > i] = 1
                print(i,np.count_nonzero(SegB), 'SegB')
                RefB = np.copy(Ref)
                RefB[Ref < 0.5] = 0
                RefB[Ref >= 0.5] = 1
                PE = perform_evaluation(SegB, RefB, SegB, 6, PixDim)
                TPc, FPc, FNc = PE.ConnectedElements()
                OER, OEFP, OEFN = PE.OE()
                DE, DEFP, DEFN = PE.DE()
                print >> result_file, '%s, %s, %f, %d, %d , %d, %d, %d, %d, %d, %d, %f, %f, %f, %f, %d, %d, %d, %d, %d, %f, %f, %f, %f, %f, %f, %f, %f' \
                                      % (nameRef, nameSeg, i, np.sum(RefB), np.sum(SegB), PE.TP(), PE.FP(), PE.FN(), TPc, FPc,
                                         FNc, PE.DSC(), PE.AvDist(),
                                         PE.VolDiff(), OER, OEFP, OEFN, DE, DEFP, DEFN, PE.PPV(), PE.NPV(), PE.FPR(),
                                         PE.Sensitivity(), PE.Specificity(), PE.Accuracy(), PE.Jaccard(), PE.HD()
                                         )





