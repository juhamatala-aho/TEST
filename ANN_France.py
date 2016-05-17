def rolling_window(data, timeFrameDict):

    inputSet=list()

    for column in data.columns:

        firstPeriod=timeFrameDict[column][0]
        lastPeriod=timeFrameDict[column][1]

        timeframe=lastPeriod-firstPeriod

        dataResult=pandas.DataFrame()

        m=len(data.index)

        for i in range(firstPeriod,m-timeframe):
            values=(data[column].iloc[i:i+timeframe].values)
            indexValue=[data[column].index[i-firstPeriod]]
            addRow=np.concatenate((indexValue,values),axis=0)

            dataResult = dataResult.append(pandas.Series(addRow), ignore_index=True)

        dataResult = dataResult.set_index(0)

        inputSet.append(dataResult)

    result = pandas.concat(inputSet, axis=1, join_axes=[inputSet[0].index],ignore_index=True)


    return result



def dataSetSetup(trainingData, targetData):

    numInputNodes=len(trainingData.columns)
    numOutputNodes=len(targetData.columns)

    dataset = SupervisedDataSet(numInputNodes, numOutputNodes)


    for i in range(0,len(targetData.index)):

        dataset.addSample(trainingData.values[i], targetData.values[i])

    return dataset


def addMonth(trainingData):

    Result=pandas.DataFrame()

    for indexI in trainingData.index:

        values = int(indexI.to_timestamp().month)
        indexValue = indexI
        addRow = np.concatenate(([indexValue], [values]), axis=0)

        Result = Result.append(pandas.Series(addRow), ignore_index=True)

    Result = Result.set_index(0)
    Result = pandas.concat((Result,trainingData), axis=1, join_axes=[trainingData[0].index], ignore_index=True)

    return Result


def outputValues(outputFrame, startValue, endValue):

    Result=pandas.DataFrame()
    period=endValue-startValue


    for i in range (startValue, len(outputFrame.index)+2-period):

        values = outputFrame.values[i-1:i-1+period]
        indexValue = outputFrame.index[i-startValue]
        addRow = np.concatenate(([indexValue], values), axis=0)
        Result = Result.append(pandas.Series(addRow), ignore_index=True)

    Result = Result.set_index(0)

    return Result


from pandasdmx import Request
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
import pandas
plt.style.use('ggplot')
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet



rawWood = '44071031+44071033+44071091+44071093+44071071+44071079+44071098+44071099'
processedWood = '44071015+44071031+44071033+44071038+44071039+44091018'
noDatarawWood = '44071010+44071030+44071039+44071050+44091011+44091019+44091090'

startPrediod='2002-02'
endPeriod = '2016-02'
country='FR'


rawWoodResp = Request('ESTAT').data('DS-016890', key={'REPORTER': country,'FLOW':'1','PARTNER':'FI','PRODUCT':rawWood,'INDICATORS':'VALUE_IN_EUROS','FREQ':'M'},
                                          params={'startPeriod': startPrediod,'endPeriod':endPeriod})


dwellingResp = Request('ESTAT').data('sts_cobp_m', key={'GEO': country, 'NACE_R2': 'F_CC112','S_ADJ':'NSA','INDIC_BT':'PNUM','UNIT':'I10','FREQ':'M'},
                                     params={'startPeriod': startPrediod, 'endPeriod':endPeriod})


#whiteWood=pandas.io.parsers.read_csv('C:\\Users\\k403055\\AnalysisData\\FR_Whitewood.csv',delimiter=';')



rawWoodData = rawWoodResp.write(rawWoodResp.data.series)
rawWoodData = rawWoodData.sort_index(ascending=True)
sumRawWoodData = rawWoodData.sum(axis=1)
sumRawWoodData -= sumRawWoodData.mean()
sumRawWoodData /= sumRawWoodData.std()

dwellingData = dwellingResp.write(dwellingResp.data.series)
dwellingData = dwellingData.sort_index(ascending=True)
dwellingData -= dwellingData.mean()
dwellingData /= dwellingData.std()

print(dwellingData)



#whiteWoodSales=whiteWood['Sales'].values
#whiteWoodSales=preprocessing.scale(whiteWoodSales)
#whiteWoodLE=whiteWood['LE'].values



trainingDataFrame = pandas.concat([sumRawWoodData, dwellingData], axis=1, join_axes=[sumRawWoodData.index])
trainingDataFrame.columns = ['rawWoodExport', 'dwelling']

timeFrameDict = {'rawWoodExport':[0,12],'dwelling':[0,8]}

trainingData=rolling_window(trainingDataFrame, timeFrameDict)
trainingData=addMonth(trainingData)


outPut=outputValues(sumRawWoodData, 13, 24)




ds2=dataSetSetup(trainingData,outPut)

#Network

inputNodes=len(trainingData.columns)
hiddenNodes=40
outputNodes=len(outPut.columns)


print(inputNodes,hiddenNodes,outputNodes)


net = buildNetwork(inputNodes, hiddenNodes, outputNodes)

trainer = BackpropTrainer(net, ds2,)
zTrain=trainingData

trainer.trainEpochs(epochs=600)

predictions=np.empty(shape=(len(ds2['target']),outputNodes))


for i in range (0,len(ds2['target'])):
    predictions[i] = net.activate(zTrain.values[i])

lastPredictions=np.empty(shape=(outputNodes,1))

lastPredictions = net.activate(zTrain.values[-10])

totalPredictions = np.concatenate((predictions[:,(outputNodes-1)],lastPredictions))

plt.plot(totalPredictions,color='blue')
plt.plot(predictions[:,outputNodes-1],color='red')
plt.plot(sumRawWoodData.values[outputNodes+12-1:],color='green')

plt.show()

writer = pandas.ExcelWriter('output.xlsx')
trainingData.to_excel(writer,'Sheet1')

writer.save()


