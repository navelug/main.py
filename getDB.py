import numpy

def isValid(data, threshold):
    conf_arr = data[2::3] # confidence levels array
    fltCnt = len([conf for conf in conf_arr if conf < 0.1]) # count number of confidence levels lower than 0.1
    if fltCnt < threshold: # determine whether this sample is an outlier
        return 1
    return 0

def getDB(gestures_num, participants_num):
    samples = []
    labels = []
    for G in range(1, gestures_num + 1): # G = 1,2,..,gestures_num
        for P in range(1, participants_num + 1): # P = 1,2,..,subjects_num
            for R in range(1, 11): # R = 1,2,..,10
                data = numpy.load('C:\\Users\\talgo\\Documents\\Python Scripts\\Tal\\Dataset\\G'+str(G)+'\\P'+str(P)+'_R'+str(R)+'.npy').tolist()
                if isValid(data, 3): # filter outliers
                    samples.append(data), labels.append(float(G) - 1)
    return samples, labels