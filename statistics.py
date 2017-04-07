import os
import os.path
import sys
import time
import scipy
import nibabel as nib
import numpy as np
import tensorflow as tf
from stats_dep import perform_statistics
#from ImageAndPatches import Volume_Sampler
from morphology import get_morphology
import util

#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def run(param):
    name_for_file = os.path.split(param.data_image_dir)[1] + '_' + os.path.split(param.seg_image_dir)[1]
    print(name_for_file)
    list_names = None
    if param.list_file == None and param.name_seg == None or (len(param.list_file)<3 and len(param.name_seg)<3):
        namesSeg = util.list_nifti_files(param.seg_image_dir)
        namesData = util.list_nifti_files(param.data_image_dir)
    elif param.name_seg is not None and len(param.name_seg)>3:
        list_names = param.name_seg
    elif param.list_file is not None:
        text_file = open("param.list_file", "r")
        list_names = text_file.read().split(',')
    if list_names is not None:
        for name in list_names:
            namesData_temp, namesSeg_temp = util.list_associations_nifti_files(param.data_image_dir, param.seg_image_dir,
                                                                              name)
            if len(namesSeg_temp) != len(namesData_temp):
                namesSeg_new = [item for item in namesSeg_temp for i in xrange(len(namesData_temp))]
                namesData_new = [namesData_temp for i in xrange(len(namesSeg_temp))]
            else:
                namesSeg_new = namesSeg_temp
                namesData_new = namesData_temp
            namesSeg.append(namesSeg_new)
            namesData.append(namesData_new)
            name_for_file = name_for_file + '_'.join(str(e) for e in list_names)


    # Different situations to be handled: 1 output file per pair of Seg and Data file. Seg can be labels, probability in 3D or probability 4D or label 4D. Data can be 3D or 4D.
    for S in namesSeg:
        NameSeg = os.path.join(param.seg_image_dir,S)
        SegNii = nib.load(NameSeg)
        SegTest = SegNii.get_data()
        for D in namesData:
            NameData = os.path.join(param.data_image_dir,D)
            DataNii = nib.load(NameData)
            Data = DataNii.get_data()
            # Create name of report file
            result_file = open(os.path.join(param.save_out_dir,os.path.basename(NameSeg).rsplit(param.ext)[0]+'_'+os.path.basename(NameData).rsplit(param.ext)[0]+'_'+param.name_out+'.csv'),'w+')
            # First line of report file to be set up according to form of Data
            StringFirstLine='Dim,Label,CoMx,CoMy,CoMz,Number,NumberBin,Vol,VolBin,SurfaceNumb,SurfaceNumbBin,SurfaceVol,SurfaceVolBin,' \
                            'SAVNumb,SAVNumbBin,SAV,SAVBin,CompactnessNumb,CompactnessNumbBin,Compactness,CompactnessBin'
            FourDValue = 1
            if Data.ndim == 4:
                FourDValue = Data.shape[3]
            StringMin = ""
            StringMax = ""
            StringAverage = ""
            StringSkewness = ""
            StringMean = ""
            Stringp25 = ""
            Stringp75 = ""
            StringKurtosis = ""
            StringMedian = ""
            StringSD = ""
            for i in range(0,FourDValue):
                StringMean = StringMean+',Mean'+str(i)
                StringMedian = StringMedian+',Median'+str(i)
                StringAverage = StringAverage+',Average'+str(i)
                StringSkewness = StringSkewness+',Skewness'+str(i)
                StringKurtosis = StringKurtosis+',Kurtosis'+str(i)
                StringMin = StringMin+',Min'+str(i)
                StringMax = StringMax+',Max'+str(i)
                Stringp25 = Stringp25+',p25'+str(i)
                Stringp75 = Stringp75+',p75'+str(i)
                StringSD = StringSD+',SD'+str(i)
            StringFirstLine=StringFirstLine+','+StringMean+StringMin+StringMax+StringSD+Stringp25+StringMedian+Stringp75\
                            +StringKurtosis+StringSkewness+StringAverage
            print >> result_file, StringFirstLine
            if SegTest.ndim == 3: # Case where segmentation is a single 3D file
                rangeDim = [0]
            else:
                rangeDim = np.arange(0,SegTest.shape[3])
            for d in rangeDim:
                if len(rangeDim)==1:
                    SegImg = SegTest
                else:
                    SegImg = np.squeeze(SegTest[...,d])
                if param.type_stats == "binary" and np.max(SegImg)>1:
                    SegImg[SegImg>=param.threshold] = 1
                    SegImg[SegImg<param.threshold] = 0
                print(np.max(SegImg),len(np.unique(SegImg)))
                if np.max(SegImg)==1 and len(np.unique(SegImg))>2:
                    type_Seg = "Probabilities"
                    print('Probabilities')
                    rangeCheck = np.arange(0, 1, float(param.step))
                elif np.max(SegImg )== 1 and len(np.unique(SegImg)) == 2:
                    type_Seg = "Binary"
                    print('Binary')
                    rangeCheck = [0.5]
                    GMSeg = get_morphology(SegImg,24)
                    Labels = GMSeg.LabelBinary()
                    rangeCheck = np.arange(1,np.max(Labels))
                else:
                    type_Seg = "Labels"
                    print('Labels analysis')
                    rangeCheck = np.arange(1,np.max(SegImg))
                for i in rangeCheck:
                    StringResults = str(d)
                    StringResults=StringResults+','+str(i)
                    if not type_Seg == "Binary":
                        seg = np.copy(SegImg)
                    else:
                        seg = np.copy(Labels)
                    print(np.max(seg))
                    seg[seg<i]=0
                    if type_Seg == "Labels" or type_Seg == "Binary":
                        seg[seg>i]=0
                        seg[seg==i]=1
                    print(np.count_nonzero(seg))
                    if np.count_nonzero(seg)>0.5:
                        Stats = perform_statistics(seg,Data,24)
                        CoM = Stats.CoM()
                        StringResults = StringResults+','.join(str(e) for e in CoM)
                        [N,Nb,V,Vb] = Stats.Vol()
                        StringResults = StringResults+','+str(N)
                        StringResults = StringResults + ',' + str(Nb)
                        StringResults = StringResults + ',' + str(V)
                        StringResults = StringResults + ',' + str(Vb)
                        [S,Sb,Sv,Svb] = Stats.Surface()
                        StringResults = StringResults + ',' + str(S)
                        StringResults = StringResults + ',' + str(Sb)
                        StringResults = StringResults + ',' + str(Sv)
                        StringResults = StringResults + ',' + str(Svb)
                        [SAV,SAVb,SAVv,SAVvb] = Stats.SAV()
                        StringResults = StringResults + ',' + str(SAV)
                        StringResults = StringResults + ',' + str(SAVb)
                        StringResults = StringResults + ',' + str(SAVv)
                        StringResults = StringResults + ',' + str(SAVvb)
                        [C,Cb,Cv,Cvb] = Stats.Compactness()
                        StringResults = StringResults + ',' + str(C)
                        StringResults = StringResults + ',' + str(Cb)
                        StringResults = StringResults + ',' + str(Cv)
                        StringResults = StringResults + ',' + str(Cvb)
                        Mean = Stats.Mean()
                        StringResults = StringResults + ',' + ','.join(str(e) for e in Mean)
                        Min = Stats.Min()
                        StringResults = StringResults + ',' + ','.join(str(e) for e in Min)
                        Max = Stats.Max()
                        StringResults = StringResults + ',' + ','.join(str(e) for e in Max)
                        SD = Stats.SD()
                        StringResults = StringResults + ',' + ','.join(str(e) for e in SD)
                        Quantiles = Stats.Quantiles()
                        if FourDValue>1:
                            for p in range(0,3):
                                StringResults = StringResults + ',' + ','.join(str(e) for e in Quantiles[:,p])
                        else:
                            StringResults + ',' + ','.join(str(e) for e in Quantiles)
                        Average = Stats.Average()
                        StringResults = StringResults + ',' + ','.join(str(e) for e in Average)
                        Skewness = Stats.Skewness()
                        StringResults = StringResults + ',' + ','.join(str(e) for e in Skewness)
                        Kurtosis = Stats.Kurtosis()
                        StringResults = StringResults + ',' + ','.join(str(e) for e in Kurtosis)
                        print >> result_file, StringResults
                    else:
                        print(i,' volume is 0')
            result_file.close()










