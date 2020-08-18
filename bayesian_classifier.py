import math
import csv, random
import math
import numpy as np
from sklearn.metrics import confusion_matrix
#from sklearn.linear_model import LinearRegression  ~to impute fuv from nuv, no longer required
#from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt



def avrg(n):
    return sum(n)/float(len(n))
    #print(zeroerror here)
def std(num):
    avg = avrg(num)
    #print(flag here)
    vrn = sum([pow(x-avg,2) for x in num])/float(len(num) -1)
    return math. sqrt(vrn)
def classdata(dst):
    cd={}
    for i in range(len(dst)):
        v=dst[i]
        if(v[-1] not in  cd):
            cd[v[-1]]=[]
        cd[v[-1]].append(v)
    return cd

def readycsv(fil):
    lines=csv.reader(open(fil,'r'))
    dst=list(lines)
    for i in range(1,len(dst)):
        dst[i]=[float(dst[i][x]) for x in range(1,len(dst[i]))]
    return dst[1:]
def splitdata(dst,r):
        ts=int(len(dst)*r)
        t_set=[]
        copy=list(dst)
        while(len(t_set)<ts):
            index=random.randrange(len(copy))
            t_set.append(copy.pop(index))
        return [t_set,copy]

def process(dst):
    foreveryclass=[]
    for attr in zip(*dst):
        x1=avrg(attr)
        y1=std(attr)
        foreveryclass.append([x1,y1])
    del foreveryclass[-1]
    return foreveryclass
def summary(dst):
    divided=classdata(dst)
    probas={}
    for cv,ins in divided.items():
        probas[cv]=process(ins)
    return    probas
def Prob(x, avrg, stdev):
    exponent = math.exp(-(math.pow(x-avrg,2)/(2*math.pow(stdev ,2))))
    return (1 / (math. sqrt(2*math.pi) * stdev)) *exponent
def ClassProb(ProcessValues , inputVector):
    probabs={}
    for classValue,classSummaries in ProcessValues.items() :
        probabs[classValue]=1
        for i in range(len(classSummaries)):
            avrg,stdev=classSummaries[i]
            x=inputVector[i]
            probabs[classValue]*=Prob(x,avrg,stdev) #print(probabs)
    return probabs
def withpriors(probabs,testset):
    st=0
    qu=0
    for i in range(len(testset)):
        if(testset[i][-1]==0):
            st = st +1
        else:
            qu = qu+1
    st = st/len(testset)
    qu = qu/len(testset)
    for j in probabs:
        if(j==0):
            probabs[j]*=st    # for log, chanfe * to + and st to math.log(abs(st))
        if(j==1):
            probabs[j]*=qu    #same here
    return probabs
def predict(ProcessValues , inputVector,testset):
    probabs = ClassProb(ProcessValues , inputVector)
    #t1=list(probabs.values())[1]
    #probabs = withpriors(probabs,testset)
    #t2=list(probabs.values())[1]
    bestLabel , bestProb = None, -1
    for classValue , probability in probabs .items() :
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
#    print(t1,t2)
    return bestLabel
def getPredictions(ProcessValues , testSet):
    predictions = []
    y_true = []
    for i in range(len(testSet)):
        result = predict(ProcessValues , testSet [ i ],testSet)
        predictions .append(result) #print(predictions)
    for i in range(len(testSet)):
        vector=testSet [ i ]
        y_true.append(vector[-1]) #print(y_true)
    return [y_true , predictions ]
def getAccuracy(testSet , predictions):
    correctn = 0
    correctp = 0
    for i in range(len(testSet)):
        if ((testSet [ i ][-1] == predictions [ i ]) and testSet[i][-1]==0):
            correctn += 1
        if((testSet [ i ][-1] == predictions [ i ]) and testSet[i][-1]==1):
            correctp += 1
    return ((correctp+correctn)/len(testSet))
def main():
    file="catalog4/cat4boost0_2.csv"     #learning on even set
    file2 = "catalog4/cat4boost1.csv"         #testing on original/unbalanced one
    file3 = "catalog4/cat4_2.csv"
    dst2=readycsv(file2)
    ratio = 0.80
    training2,test2=splitdata(dst2,ratio)
    dst=readycsv(file)
    training,test=splitdata(dst,ratio)
    pv=summary(training)
    y_true,pred=getPredictions(pv,test2)
    cm=confusion_matrix(y_true,pred)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_true,pred).ravel()
    print(tp1,fn1,"\n",fp1,tn1)
    #print( '\n'. join ([ ''. join ([ '{:4} '.format(item) for item in row]) for row in cm])) #confusionmatrix = np.matrix(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    #FP,TP,FN,TN = fp1,tp1,fn1,tn1
    TPR = TP/(TP+FN)
    
    TNR = TN/(TN+FP)
    Precision = TP/(TP+FP)
    print("Precision",Precision)
    
    Recall = TP/(TP+FN)
    print("r",Recall)
    Acc = (TP+TN)/(TP+TN+FP+FN)
    coin = [1]*len(test2)
    #m1 = getAccuracy(test,pred)
    m1 = getAccuracy(test2,coin)  #check how much better than coin toss/ blind classification
    print("Blindness/ Skew",m1)
    print("Accuracy", Acc*100)
    Fscore = 2*(Precision*Recall)/(Precision+Recall)
    print("Fscore", Fscore)
    plt.plot([0.37,0.45,0.52,0.59,0.62,0.76,0.89],[0.46,0.24,0.22,0.31,0.41,0.47,0.55])
plt.ylabel('Diff F_scores')
plt.xlabel("Blindness")
plt.annotate("60.98", # this is the text
                 (0.45,0.24), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 ha='center')
plt.annotate("71.1", # this is the text
                 (0.59,0.31), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 ha='center')
plt.annotate("73", # this is the text
                 (0.37,0.46), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 ha='center')
plt.annotate("61", # this is the text
                 (0.52,0.22), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 ha='center')
plt.annotate("73", # this is the text
                 (0.76,0.47), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,5), # distance from text to points (x,y)
                 ha='center')
plt.annotate("72", # this is the text
                 (0.62,0.41), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center')
plt.annotate("77", # this is the text
                 (0.89,0.55), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center')
plt.show
#    file2 = 'catalog3/cat3.csv'
#    df = pd.read_csv(file2)
#    x2 = np.array(df.nuv_mag)
#    y2 = np.array(df.fuv_mag)
#    x2, y2 = np.array(x2), np.array(y2)
#    x_=np.reshape(x2,(-1,1))

   
#    model = LinearRegression().fit(x_, y2)

#    r_sq = model.score(x_, y2)
#    intercept, coefficients = model.intercept_, model.coef_
#    print(intercept, coefficients)
main()



