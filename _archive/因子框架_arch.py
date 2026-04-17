# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:07:20 2021

@author: 13121
"""

#In[1] 导入模块
import numpy as np
import pandas as pd
import math
from pandas.core import resample
import copy
import time
from sklearn.linear_model import LinearRegression


#In[2] 数据处理
#将收盘价、开盘价、是否ST、交易状态、上市交易日数数据读入
path="E:\\python文档\\"
def dataread(filename):
    data=pd.read_csv(filename,index_col=0)
    data.drop("日期",axis=1,inplace=True,errors="ignore")
    data.index.names=["股票代码"]
    data.columns.names=["日期"]
    data=data.T
    data=data[(data.index>="20140101")&(data.index<="20201231")]
    data.index=pd.to_datetime(data.index)
    return data
stockOpening=dataread(path+"后复权开盘价.csv")
stockClosing=dataread(path+"后复权收盘价.csv")
stockST=dataread(path+"是否ST股.csv")
stockTradingStatus=dataread(path+"交易状态.csv")
stockTradingDays=dataread(path+"上市交易日数.csv")
stockTurn=dataread(path+"换手率.csv")
stockPB=dataread(path+"PB.csv")
stockClosing_market=dataread(path+"不复权收盘价.csv")
Flowofequity=dataread(path+"流通股本.csv")
#数据预处理
stockOpening[stockOpening==0]=stockClosing[stockOpening==0]


#In[3] 计算反转因子
#生成一个判断矩阵，满足ST、停牌股和次新股三个条件其中之一的日内收盘价将其值设为nan
def stocksprefilter(stocks,stocks1,stocks2,stocks3):
    stocks_prefilter=copy.deepcopy(stocks)
    stocks_prefilter[(stocks1.values==1)|(stocks2.values!="交易")|(stocks3.values<60)]=np.nan
    return stocks_prefilter
stockClosing_prefilter=stocksprefilter(stockClosing,stockST,stockTradingStatus,stockTradingDays)
#生成一个回看选择矩阵，将其不满足数据有效性条件的股票日内收盘价设为nan
def stocksfilter(stocks,Ret,vaildnum):
    stocks_filter=copy.deepcopy(stocks)
    counts=stocks_filter.rolling(window=Ret).count()
    stocks_filter[counts<Ret*vaildnum]=np.nan  
    return stocks_filter
stockClosing_filter=stocksfilter(stockClosing_prefilter,Ret=20,vaildnum=0.5)
#计算每月月底回看过去20个交易日各股票的累计收益率即因子值
def stocksscore(stocks,Ret):
    stocks_score=stocks/stocks.shift(Ret)-1
    stocks_score=stocks_score.resample("M").last()
    return stocks_score
Ret20=stocksscore(stockClosing_filter,Ret=20)


#In[3] 计算换手率因子
#生成一个判断矩阵，满足ST、停牌股和次新股三个条件其中之一的日内收盘价将其值设为nan
def stocksprefilter(stocks,stocks1,stocks2,stocks3):
    stocks_prefilter=copy.deepcopy(stocks)
    stocks_prefilter[(stocks1.values==1)|(stocks2.values!="交易")|(stocks3.values<60)]=np.nan
    return stocks_prefilter
stockTurn_prefilter=stocksprefilter(stockTurn,stockST,stockTradingStatus,stockTradingDays)
#生成一个回看选择矩阵，将其不满足数据有效性条件的股票日内收盘价设为nan
def stocksfilter(stocks,Ret,vaildnum):
    stocks_filter=copy.deepcopy(stocks)
    counts=stocks_filter.rolling(window=Ret).count()
    stocks_filter[counts<Ret*vaildnum]=np.nan  
    return stocks_filter
stockTurn_filter=stocksfilter(stockTurn_prefilter,Ret=20,vaildnum=0.5)
#计算每月月底回看过去20个交易日各股票的累计收益率即因子值
def stocksscore(stocks,Ret):
    stocks_score=stocks.rolling(window=Ret).mean()
    stocks_score=stocks_score.resample("M").last()
    return stocks_score
Turn20=stocksscore(stockTurn_filter,Ret=20)
#进行市值中性化，排除市值的影响和干扰
stockClosing_market_month=stockClosing_market.resample("M").last()
Flowofequity_month=Flowofequity.resample("M").last()
stocksMarketValue=np.log(stockClosing_market_month*Flowofequity_month)
Turn20_desize=pd.DataFrame(columns=stocksMarketValue.columns)
for indexi in stocksMarketValue.index[1:]:
    stocks=pd.concat([Turn20.loc[indexi],stocksMarketValue.loc[indexi]],axis=1)    
    stocks.columns=["换手率","市值"]
    stocks.replace([np.inf,-np.inf],np.nan,inplace=True)
    stocks.dropna(axis=0,how="any",inplace=True)
    Turn20_model=LinearRegression()
    Turn20_model.fit(stocks["市值"].values.reshape(-1,1),stocks["换手率"].values)    
    stocks["换手率_desize"]=stocks["换手率"]-Turn20_model.predict(stocks["市值"].values.reshape(-1,1))    
    Turn20_desize=Turn20_desize.append(stocks["换手率_desize"])
Turn20_desize.index=stocksMarketValue.index[1:] 
Turn20_desize=Turn20_desize.reindex(index=stocksMarketValue.index)    


#In[3] 计算综合因子：换手率+反转
RetandTurn=(Ret20.values-np.nanmean(Ret20,axis=1).reshape(-1,1))/np.nanstd(Ret20,axis=1).reshape(-1,1)\
            +(Turn20_desize.values-np.nanmean(Turn20_desize,axis=1).reshape(-1,1))/np.nanstd(Turn20_desize,axis=1).reshape(-1,1)
RetandTurn=pd.DataFrame(RetandTurn,index=Ret20.index,columns=Ret20.columns)


#In[3] 计算PB因子
#生成一个判断矩阵，满足ST、停牌股和次新股三个条件其中之一的日内收盘价将其值设为nan
def stocksprefilter(stocks,stocks1,stocks2,stocks3):
    stocks_prefilter=copy.deepcopy(stocks)
    stocks_prefilter[(stocks1.values==1)|(stocks2.values!="交易")|(stocks3.values<60)]=np.nan
    return stocks_prefilter
stockPB_prefilter=stocksprefilter(stockPB,stockST,stockTradingStatus,stockTradingDays)
#生成一个回看选择矩阵，将其不满足数据有效性条件的股票日内收盘价设为nan
def stocksfilter(stocks,Ret,vaildnum):
    stocks_filter=copy.deepcopy(stocks)
    counts=stocks_filter.rolling(window=Ret).count()
    stocks_filter[counts<Ret*vaildnum]=np.nan  
    return stocks_filter
stockPB_filter=stocksfilter(stockPB_prefilter,Ret=20,vaildnum=0.5)
#计算每月月底回看过去20个交易日各股票的累计收益率即因子值
def stocksscore(stocks,Ret):
    stocks_score=stocks.rolling(window=Ret).mean()
    stocks_score=stocks_score.resample("M").last()
    return stocks_score
PBpre20=stocksscore(stockPB_filter,Ret=20)
#极值处理
delmatrix1=pd.DataFrame(np.tile((np.nanmean(PBpre20,axis=1)+3*np.nanstd(PBpre20,axis=1)).reshape(-1,1),4266),index=PBpre20.index,columns=PBpre20.columns)
delmatrix2=pd.DataFrame(np.tile((np.nanmean(PBpre20,axis=1)-3*np.nanstd(PBpre20,axis=1)).reshape(-1,1),4266),index=PBpre20.index,columns=PBpre20.columns)
#3倍标准差以外的值变成nan
def stocksdel_1(stocks,stocks1,stocks2):
    stocks[(stocks>stocks1)|(stocks<stocks2)]=np.nan
    return stocks
PB20_1=stocksdel_1(PBpre20,delmatrix1,delmatrix2)
#3倍标准差以外的值拉回
def stocksdel_2(stocks,stocks1,stocks2):
    stocks[stocks>stocks1]=stocks1
    stocks[stocks<stocks2]=stocks2
    return stocks
PB20_2=stocksdel_2(PBpre20,delmatrix1,delmatrix2)
#进行市值和行业中性化(运算需要时间，大概45分钟左右)
stockClosing_market_month=stockClosing_market.resample("M").last()
Flowofequity_month=Flowofequity.resample("M").last()
stocksMarketValue=np.log(stockClosing_market_month*Flowofequity_month)
storeIndustry=pd.HDFStore("FactorLoading_Industry.h5")
PB20_desize1=pd.DataFrame(columns=stocksMarketValue.columns)
for indexi in stocksMarketValue.index[1:]:
    stocks=pd.concat([PB20_1.loc[indexi],stocksMarketValue.loc[indexi]],axis=1)    
    stocks.columns=["PB","市值"]
    for keyi in storeIndustry.keys():
        storei=storeIndustry[keyi]
        storei.index=pd.to_datetime(storei.index)
        storei=storei.resample("M").last()
        stocks[keyi]=storei.loc[indexi]
    stocks.replace([np.inf,-np.inf],np.nan,inplace=True)
    stocks.dropna(axis=0,how="any",inplace=True)
    PB20_model=LinearRegression()
    PB20_model.fit(stocks.iloc[:,1:],stocks["PB"])    
    stocks["PB_desize"]=stocks["PB"]-PB20_model.predict(stocks.iloc[:,1:])    
    PB20_desize1=PB20_desize1.append(stocks["PB_desize"])
PB20_desize1.index=stocksMarketValue.index[1:] 
PB20_desize1=PB20_desize1.reindex(index=stocksMarketValue.index)    
PB20_desize2=pd.DataFrame(columns=stocksMarketValue.columns)
for indexi in stocksMarketValue.index[1:]:
    stocks=pd.concat([PB20_2.loc[indexi],stocksMarketValue.loc[indexi]],axis=1)    
    stocks.columns=["PB","市值"]
    for keyi in storeIndustry.keys():
        storei=storeIndustry[keyi]
        storei.index=pd.to_datetime(storei.index)
        storei=storei.resample("M").last()
        stocks[keyi]=storei.loc[indexi]
    stocks.replace([np.inf,-np.inf],np.nan,inplace=True)
    stocks.dropna(axis=0,how="any",inplace=True)
    PB20_model=LinearRegression()
    PB20_model.fit(stocks.iloc[:,1:],stocks["PB"])    
    stocks["PB_desize"]=stocks["PB"]-PB20_model.predict(stocks.iloc[:,1:])    
    PB20_desize2=PB20_desize2.append(stocks["PB_desize"])
PB20_desize2.index=stocksMarketValue.index[1:] 
PB20_desize2=PB20_desize2.reindex(index=stocksMarketValue.index)    


#In[4] 因子回测

classnum=5   #参数可调，5或者10
factor=PB20_desize2   #因子可换，Ret20或Turn20_desize或RetandTurn或PB20_desize1或PB20_desize2

#计算各股票的月收益率
stockClosing_return=stockClosing.resample("M").last()/stockOpening.resample("M").first()-1
stockClosing_return[np.isinf(stockClosing_return)]=np.nan  
#月初涨停股和月末跌停股
stocks_rlimopening=(stockOpening/stockClosing.shift(1)-1).resample("M").first()
stocks_rlimopening[np.isinf(stocks_rlimopening)]=np.nan
stocks_limdclosing=(stockClosing/stockClosing.shift(1)-1).resample("M").last()
stocks_limdclosing[np.isinf(stocks_limdclosing)]=np.nan
#月初涨停股不买入
stockClosing_return[stocks_rlimopening>0.098]=np.nan    
#计算分组收益率
stocks_factoranalysis=pd.DataFrame()
limdownstocks=pd.DataFrame()
for indexi in stockClosing_return.index:
    stocks_scorereturn=pd.concat([factor.shift(1).loc[indexi,:],stockClosing_return.loc[indexi,:]],axis=1)    
    stocks_scorereturn.columns=["因子值","月收益率"]
    stocks_scorereturn.dropna(axis=0,how="any",inplace=True)
    stocks_scorereturn=stocks_scorereturn.sort_values(by=["因子值"])
    stocksnum=stocks_scorereturn["因子值"].count()
    stocks_scorereturn["class"]=(np.arange(stocksnum)+1)//(stocksnum//classnum)
    stocks_scorereturn.loc[stocks_scorereturn["class"]==classnum,"class"]=classnum-1
    stocks_scorereturn_append=stocks_scorereturn.append(limdownstocks)
    limdownstocks=stocks_scorereturn.loc[stocks_limdclosing.loc[indexi,:]<-0.098,:]
    stocks_factoranalysis=stocks_factoranalysis.append(stocks_scorereturn_append.groupby("class")["月收益率"].mean())
stocks_factoranalysis.index=stockClosing_return.index
columns1=["组别1","组别2","组别3","组别4","组别5","组别6","组别7","组别8","组别9","组别10"]
stocks_factoranalysis.columns=columns1[:classnum]
#计算多空对冲绩效指标（hedgmreturn：多空对冲收益；hedgcumvalue：多空对冲净值；hedgyearR：年化收益；hedgyearsigma：年化波动；informratio：收益波动比；winratio：月度胜率；drawdown：最大回撤）
def stockshedge(stocks):
    hedgmreturn=stocks.iloc[:,0]-stocks.iloc[:,-1]
    hedgcumvalue=(hedgmreturn+1).cumprod()
    hedgyearR=hedgcumvalue[-1]**(12/hedgmreturn.count())-1
    hedgyearsigma=hedgmreturn.std()*math.sqrt(12)
    informratio=hedgyearR/hedgyearsigma   
    winratio=hedgmreturn[hedgmreturn>0].count()/hedgmreturn.count()
    drawmonth=pd.Series(index=hedgcumvalue.index)
    for indexi in drawmonth.index:
        drawmonthi=hedgcumvalue[indexi]/hedgcumvalue[:indexi].max()-1
        drawmonth[indexi]=drawmonthi
    drawdown=-drawmonth.min()
    return [hedgmreturn,hedgcumvalue,hedgyearR,hedgyearsigma,informratio,winratio,drawdown]
[hedgmreturn,hedgcumvalue,hedgyearR,hedgyearsigma,informratio,winratio,drawdown]=stockshedge(stocks_factoranalysis)
#计算IC指标（IC：相关系数；ICave：IC均值；ICIR:年化ICIR；RankIC：秩相关系数；RankICave：RankIC均值；RankICIR：年化RankICIR）
def stocksfactorreturn(stocks1,stocks2):
    stocks=pd.concat([stocks1.shift(1).stack(),stocks2.stack()],axis=1)
    stocks.columns=["因子值","月收益率"]
    stocks.dropna(axis=0,how="any",inplace=True)
    stocks=stocks.sort_values(by=["日期","因子值"])
    return stocks
stocks_factorreturn=stocksfactorreturn(factor,stockClosing_return)
def factorIC(stocks):
    IC=stocks.groupby("日期").corr("pearson").iloc[::2,1].reset_index()
    IC=pd.Series(IC.iloc[:,-1].values,index=IC.iloc[:,0])
    ICave=IC.mean()
    ICIR=(IC.mean()/IC.std())*math.sqrt(12)
    RankIC=stocks.groupby("日期").corr("spearman").iloc[::2,1].reset_index()
    RankIC=pd.Series(RankIC.iloc[:,-1].values,index=RankIC.iloc[:,0])
    RankICave=RankIC.mean()
    RankICIR=(RankIC.mean()/RankIC.std())*math.sqrt(12)
    return [IC,ICave,ICIR,RankIC,RankICave,RankICIR]
[IC,ICave,ICIR,RankIC,RankICave,RankICIR]=factorIC(stocks_factorreturn)
 

#In[5] 归属母公司净利润因子构建框架
path="E:\\python文档\\"
def dataread(filename):
    data=pd.read_csv(filename,index_col=0)
    data.drop("日期",axis=1,inplace=True,errors="ignore")
    data.index.names=["股票代码"]
    data.columns.names=["日期"]
    data=data.T
    data=data[(data.index>="20060101")&(data.index<="20201031")]
    data.index=pd.to_datetime(data.index)
    return data
stockOpening=dataread(path+"后复权开盘价.csv")
stockClosing=dataread(path+"后复权收盘价.csv")
stockST=dataread(path+"是否ST股.csv")
stockTradingStatus=dataread(path+"交易状态.csv")
stockTradingDays=dataread(path+"上市交易日数.csv")
stockNetProfit=dataread(path+"归属母公司净利润.csv")
stockClosing_market=dataread(path+"不复权收盘价.csv")
Flowofequity=dataread(path+"流通股本.csv")

#删除次新股、ST股和停牌股样本
stockNetProfit=stockNetProfit.resample("M").last().dropna()
stockST=stockST.resample("M").last().reindex(index=stockNetProfit.index)
stockTradingStatus=stockTradingStatus.resample("M").last().reindex(index=stockNetProfit.index)
stockTradingDays=stockTradingDays.resample("M").last().reindex(index=stockNetProfit.index)
def stocksprefilter(stocks,stocks1,stocks2,stocks3):
    stocks_prefilter=copy.deepcopy(stocks)
    stocks_prefilter[(stocks1.values==1)|(stocks2.values!="交易")|(stocks3.values<60)]=np.nan
    return stocks_prefilter
stockNP_prefilter=stocksprefilter(stockNetProfit,stockST,stockTradingStatus,stockTradingDays)
#计算净利润同比增长率
RateofRise=stockNP_prefilter/stockNetProfit.shift(3)
#极值处理
delmatrix1=pd.DataFrame(np.tile((np.nanmean(RateofRise,axis=1)+3*np.nanstd(RateofRise,axis=1)).reshape(-1,1),4266),index=RateofRise.index,columns=RateofRise.columns)
delmatrix2=pd.DataFrame(np.tile((np.nanmean(RateofRise,axis=1)-3*np.nanstd(RateofRise,axis=1)).reshape(-1,1),4266),index=RateofRise.index,columns=RateofRise.columns)
#3倍标准差以外的值变成nan
def stocksdel_1(stocks,stocks1,stocks2):
    stocks[(stocks>stocks1)|(stocks<stocks2)]=np.nan
    return stocks
ROR_1=stocksdel_1(RateofRise,delmatrix1,delmatrix2)
#3倍标准差以外的值拉回
def stocksdel_2(stocks,stocks1,stocks2):
    stocks[stocks>stocks1]=stocks1
    stocks[stocks<stocks2]=stocks2
    return stocks
ROR_2=stocksdel_2(RateofRise,delmatrix1,delmatrix2)
#进行市值中性化
stockClosing_market_month=stockClosing_market.resample("M").last().reindex(index=stockNetProfit.index)
Flowofequity_month=Flowofequity.resample("M").last().reindex(index=stockNetProfit.index)
stocksMarketValue=np.log(stockClosing_market_month*Flowofequity_month)
ROR_de1=pd.DataFrame(columns=stocksMarketValue.columns)
for indexi in stocksMarketValue.index[3:]:
    stocks=pd.concat([ROR_1.loc[indexi],stocksMarketValue.loc[indexi]],axis=1)    
    stocks.columns=["ROR","市值"]
    stocks.replace([np.inf,-np.inf],np.nan,inplace=True)
    stocks.dropna(axis=0,how="any",inplace=True)
    ROR_model=LinearRegression()
    ROR_model.fit(stocks.iloc[:,1:],stocks["ROR"])    
    stocks["ROR_desize"]=stocks["ROR"]-ROR_model.predict(stocks.iloc[:,1:])    
    ROR_de1=ROR_de1.append(stocks["ROR_desize"])
ROR_de1.index=stocksMarketValue.index[3:] 
ROR_de1=ROR_de1.reindex(index=stocksMarketValue.index)
ROR_de2=pd.DataFrame(columns=stocksMarketValue.columns)
for indexi in stocksMarketValue.index[3:]:
    stocks=pd.concat([ROR_2.loc[indexi],stocksMarketValue.loc[indexi]],axis=1)    
    stocks.columns=["ROR","市值"]
    stocks.replace([np.inf,-np.inf],np.nan,inplace=True)
    stocks.dropna(axis=0,how="any",inplace=True)
    ROR_model=LinearRegression()
    ROR_model.fit(stocks.iloc[:,1:],stocks["ROR"])    
    stocks["ROR_desize"]=stocks["ROR"]-ROR_model.predict(stocks.iloc[:,1:])    
    ROR_de2=ROR_de2.append(stocks["ROR_desize"])
ROR_de2.index=stocksMarketValue.index[3:] 
ROR_de2=ROR_de2.reindex(index=stocksMarketValue.index)    
#进行市值和行业中性化(运算需要时间，大概45分钟左右)
stockClosing_market_month=stockClosing_market.resample("M").last().reindex(index=stockNetProfit.index)
Flowofequity_month=Flowofequity.resample("M").last().reindex(index=stockNetProfit.index)
stocksMarketValue=np.log(stockClosing_market_month*Flowofequity_month)
storeIndustry=pd.HDFStore("FactorLoading_Industry.h5")
ROR_desize1=pd.DataFrame(columns=stocksMarketValue.columns)
for indexi in stocksMarketValue.index[3:]:
    stocks=pd.concat([ROR_1.loc[indexi],stocksMarketValue.loc[indexi]],axis=1)    
    stocks.columns=["ROR","市值"]
    for keyi in storeIndustry.keys():
        storei=storeIndustry[keyi]
        storei.index=pd.to_datetime(storei.index)
        storei=storei.resample("M").last()
        stocks[keyi]=storei.loc[indexi]
    stocks.replace([np.inf,-np.inf],np.nan,inplace=True)
    stocks.dropna(axis=0,how="any",inplace=True)
    ROR_model=LinearRegression()
    ROR_model.fit(stocks.iloc[:,1:],stocks["ROR"])    
    stocks["ROR_desize"]=stocks["ROR"]-ROR_model.predict(stocks.iloc[:,1:])    
    ROR_desize1=ROR_desize1.append(stocks["ROR_desize"])
ROR_desize1.index=stocksMarketValue.index[3:] 
ROR_desize1=ROR_desize1.reindex(index=stocksMarketValue.index)    
ROR_desize2=pd.DataFrame(columns=stocksMarketValue.columns)
for indexi in stocksMarketValue.index[3:]:
    stocks=pd.concat([ROR_2.loc[indexi],stocksMarketValue.loc[indexi]],axis=1)    
    stocks.columns=["ROR","市值"]
    for keyi in storeIndustry.keys():
        storei=storeIndustry[keyi]
        storei.index=pd.to_datetime(storei.index)
        storei=storei.resample("M").last()
        stocks[keyi]=storei.loc[indexi]
    stocks.replace([np.inf,-np.inf],np.nan,inplace=True)
    stocks.dropna(axis=0,how="any",inplace=True)
    ROR_model=LinearRegression()
    ROR_model.fit(stocks.iloc[:,1:],stocks["ROR"])    
    stocks["ROR_desize"]=stocks["ROR"]-ROR_model.predict(stocks.iloc[:,1:])    
    ROR_desize2=ROR_desize2.append(stocks["ROR_desize"])
ROR_desize2.index=stocksMarketValue.index[3:] 
ROR_desize2=ROR_desize2.reindex(index=stocksMarketValue.index)   


classnum=5   #参数可调，5或者10
factor=ROR_de1    #因子可换，ROR_1或ROR_2或ROR_de1或ROR_de2或ROR_desize1或ROR_desize2

#计算各股票的月收益率
stockClosing_return=(stockClosing.resample("M").last().reindex(index=stockNetProfit.index))/(stockOpening.resample("M").first().shift(-1).reindex(index=stockNetProfit.index).shift(1))-1
stockClosing_return[np.isinf(stockClosing_return)]=np.nan  
#月初涨停股和月末跌停股
stocks_rlimopening=(stockOpening/stockClosing.shift(1)-1).resample("M").first().shift(-1).reindex(index=stockNetProfit.index).shift(1)
stocks_rlimopening[np.isinf(stocks_rlimopening)]=np.nan
stocks_limdclosing=(stockClosing/stockClosing.shift(1)-1).resample("M").last().reindex(index=stockNetProfit.index)
stocks_limdclosing[np.isinf(stocks_limdclosing)]=np.nan
#月初涨停股不买入
stockClosing_return[stocks_rlimopening>0.098]=np.nan    
#计算分组收益率
stocks_factoranalysis=pd.DataFrame()
limdownstocks=pd.DataFrame()
for indexi in stockClosing_return.index:
    stocks_scorereturn=pd.concat([factor.shift(1).loc[indexi,:],stockClosing_return.loc[indexi,:]],axis=1)    
    stocks_scorereturn.columns=["因子值","月收益率"]
    stocks_scorereturn.dropna(axis=0,how="any",inplace=True)
    stocks_scorereturn=stocks_scorereturn.sort_values(by=["因子值"])
    stocksnum=stocks_scorereturn["因子值"].count()
    stocks_scorereturn["class"]=(np.arange(stocksnum)+1)//(stocksnum//classnum)
    stocks_scorereturn.loc[stocks_scorereturn["class"]==classnum,"class"]=classnum-1
    stocks_scorereturn_append=stocks_scorereturn.append(limdownstocks)
    limdownstocks=stocks_scorereturn.loc[stocks_limdclosing.loc[indexi,:]<-0.098,:]
    stocks_factoranalysis=stocks_factoranalysis.append(stocks_scorereturn_append.groupby("class")["月收益率"].mean())
stocks_factoranalysis.index=stockClosing_return.index
columns1=["组别1","组别2","组别3","组别4","组别5","组别6","组别7","组别8","组别9","组别10"]
stocks_factoranalysis.columns=columns1[:classnum]
#计算多空对冲绩效指标（一年回测3次）
def stockshedge(stocks):
    hedgmreturn=stocks.iloc[:,-1]-stocks.iloc[:,0]     #分组5是增长率大的因子，分组1是增长率小的因子，分组5对冲分组1
    hedgcumvalue=(hedgmreturn+1).cumprod()
    hedgyearR=hedgcumvalue[-1]**(3/hedgmreturn.count())-1
    hedgyearsigma=hedgmreturn.std()*math.sqrt(3)
    informratio=hedgyearR/hedgyearsigma   
    winratio=hedgmreturn[hedgmreturn>0].count()/hedgmreturn.count()
    drawmonth=pd.Series(index=hedgcumvalue.index)
    for indexi in drawmonth.index:
        drawmonthi=hedgcumvalue[indexi]/hedgcumvalue[:indexi].max()-1
        drawmonth[indexi]=drawmonthi
    drawdown=-drawmonth.min()
    return [hedgmreturn,hedgcumvalue,hedgyearR,hedgyearsigma,informratio,winratio,drawdown]
[hedgmreturn,hedgcumvalue,hedgyearR,hedgyearsigma,informratio,winratio,drawdown]=stockshedge(stocks_factoranalysis)
#计算IC指标（一年回测3次）
def stocksfactorreturn(stocks1,stocks2):
    stocks=pd.concat([stocks1.shift(1).stack(),stocks2.stack()],axis=1)
    stocks.columns=["因子值","月收益率"]
    stocks.dropna(axis=0,how="any",inplace=True)
    stocks=stocks.sort_values(by=["日期","因子值"])
    return stocks
stocks_factorreturn=stocksfactorreturn(factor,stockClosing_return)
def factorIC(stocks):
    IC=stocks.groupby("日期").corr("pearson").iloc[::2,1].reset_index()
    IC=pd.Series(IC.iloc[:,-1].values,index=IC.iloc[:,0])
    ICave=IC.mean()
    ICIR=(IC.mean()/IC.std())*math.sqrt(3)
    RankIC=stocks.groupby("日期").corr("spearman").iloc[::2,1].reset_index()
    RankIC=pd.Series(RankIC.iloc[:,-1].values,index=RankIC.iloc[:,0])
    RankICave=RankIC.mean()
    RankICIR=(RankIC.mean()/RankIC.std())*math.sqrt(3)
    return [IC,ICave,ICIR,RankIC,RankICave,RankICIR]
[IC,ICave,ICIR,RankIC,RankICave,RankICIR]=factorIC(stocks_factorreturn)
