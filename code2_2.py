
# coding: utf-8

import pandas as pd
import numpy as np
from hmmlearn import hmm
import heapq

#行业轮动匹配所需函数
def isIn(a,b,data):
    top_3=[]
    for i in range(3):
        top_3.append(data.iloc[b,i])
    if data.iloc[a,0] in top_3:
        return True
    else:
        return False

def main():
    hq=pd.read_csv("/home/shaoguang/anaconda3/shaoguang123/data_proc/hq2.csv",encoding="utf-8")
    cyb1=pd.read_csv("/home/shaoguang/anaconda3/shaoguang123/data_proc/cyb1.csv",encoding="utf-8")
    cyb2=pd.read_csv("/home/shaoguang/anaconda3/shaoguang123/data_proc/cyb2.csv",encoding="utf-8")
    hs300a=pd.read_csv("/home/shaoguang/anaconda3/shaoguang123/data_proc/hs300a.csv",encoding="utf-8")
    hs300b=pd.read_csv("/home/shaoguang/anaconda3/shaoguang123/data_proc/hs300b.csv",encoding="utf-8")
    
    data=pd.DataFrame()
    data.insert(0,"ChgPct_cyb",cyb2.iloc[:,3])
    data.insert(1,"RSI_cyb",cyb2.iloc[:,20])
    data.insert(2,"BIAS_cyb",0)
    data.insert(3,"ATR_cyb",0)
    
    close=cyb2.iloc[:,4]
    ma5=cyb2.iloc[:,8]
    for i in range(len(data.iloc[:,0])):
        data.iloc[i,2]=(close[i]-ma5[i])/ma5[i]*100
    
    L=cyb1.iloc[:,8]
    H=cyb1.iloc[:,9]
    CL=cyb1.iloc[:,10]
    for i in range(len(data.iloc[:,0])):
        if i==0 :
            data.iloc[i,3]=H[i]-L[i]
        else:
            data.iloc[i,3]=max((H[i]-L[i]),(H[i]-CL[i]),(CL[i]-L[i]))
    
    data.insert(0,"ChgPct_hs300",hs300b.iloc[:,3])
    data.insert(1,"RSI_hs300",hs300b.iloc[:,20])
    data.insert(2,"BIAS_hs300",0)
    data.insert(3,"ATR_hs300",0)
    
    close=hs300b.iloc[:,4]
    ma5=hs300b.iloc[:,8]
    for i in range(len(data.iloc[:,0])):
        data.iloc[i,2]=(close[i]-ma5[i])/ma5[i]*100
    
    L=hs300a.iloc[:,8]
    H=hs300a.iloc[:,9]
    CL=hs300a.iloc[:,10]
    for i in range(len(data.iloc[:,0])):
        if i==0 :
            data.iloc[i,3]=H[i]-L[i]
        else:
            data.iloc[i,3]=max((H[i]-L[i]),(H[i]-CL[i]),(CL[i]-L[i]))
    
    #尝试各维分别归一化，正则化,扩充特征向量，进行PCA降维处理
    
    
    #data=preprocessing.scale(data)
    #data=preprocessing.normalize(data, norm='l2')
    #pca=PCA(n_components="mle",copy=False)
    #data=pca.fit_transform(data)
    #data[0]
    
    #求得每日各行业板块涨幅 
    hq1=hq.iloc[:,[0,1,2,4,5]]
    col=int(hq1.iloc[:,0:1].size)
    dat=int(col/28)
    
    val=np.zeros((dat,28))
    row_list=np.zeros(28)
    Column_list=np.arange(0,dat,1)
    for i in range(28):
        row_list[i]=hq1.iloc[i*dat,0]
    for i in range(dat):
        for j in range(28):
            val[dat-i-1][j]=hq1.iloc[i+dat*j,4]
           
    data_tmp=pd.DataFrame(val,index=Column_list,columns=row_list)
    
    #求取每日涨幅排行前5的行业
    order1=[]
    order2=[]
    order3=[]
    order4=[]
    order5=[]
    for i in range(dat):
        Sector_list=[]
        for j in range(28):
            Sector_list.append((row_list[j],val[i][j]))
        top = heapq.nlargest(5,Sector_list,key=lambda s: s[1])
        order1.append(top[0][0]),order2.append(top[1][0]),order3.append(top[2][0]),order4.append(top[3][0]),order5.append(top[4][0])
    order_list=pd.DataFrame()
    order_list.insert(0,"order1",order1),order_list.insert(1,"order2",order2),order_list.insert(2,"order3",order3),order_list.insert(3,"order4",order4),order_list.insert(4,"order5",order5)
    
    
    #参数设置
    n=3 #隐状态数目
    T=350 #样本窗口大小
    t=1 #预测天数
    w_n=5 #与当前交易日相同市场隐含状态相同行业轮动特征且似然值最接近的天数
    index = 0
    step = t
    win=0
    lose=0
    win0=0
    win1=0
    win2=0
    win3=0
    win4=0
    win5=0
    win6=0
    
    while index+T < len(data)-step:
        model = hmm.GaussianHMM(n_components= n, covariance_type="full", n_iter=1000).fit(data[index:index+T])
        hist_info = [] 
        hiddenStatus = model.predict(data[index:index+T])
        #print(hiddenStatus)
        
        for i in range(index, index+T):
            score = model.score(data[i: i+1])
            day_tuple = (i, hiddenStatus[i-index], score)
            hist_info.append(day_tuple) 
            
        last_hiddenStatus = hist_info[-1][1]
        last_score = hist_info[-1][2]
        last_index = hist_info[-1][0]
        print(last_index)
        
        sameStatus = []
        cnt=0
        for (x,y,z) in hist_info[:-1]:
            if y == last_hiddenStatus:#市场隐含状态匹配
                if  isIn(last_index,x,order_list) and isIn(last_index-1,x-1,order_list):#行业轮动特征匹配
                    diff = abs(z - last_score)
                    sameStatus.append((x, diff))
                    cnt+=1
        if(cnt<w_n): 
            index += step
            continue
        pos_diffs = heapq.nsmallest(w_n, sameStatus, key=lambda s: s[1])
        
        #加权预测
        weights = [5,4,3,2,1]    
        d={}
        for i in range(w_n):
            if order_list.iloc[pos_diffs[i][0]+1,0] in d:
                d[order_list.iloc[pos_diffs[i][0]+1,0]]+=weights[i]/3
            else:
                d[order_list.iloc[pos_diffs[i][0]+1,0]]=weights[i]/3
                    
        for i in range(w_n):
            for j in range(1,3):
                if order_list.iloc[pos_diffs[i][0]+1,j] in d:
                    d[order_list.iloc[pos_diffs[i][0]+1,j]]+=weights[i]*(3-j)/6
        d=sorted(d.items(),key = lambda asd:asd[1],reverse=True)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(d)
        
        
        top_5=[]
        for i in range(5):
            top_5.append(order_list.iloc[last_index+1,i])
        print("####################")
        print(last_index)
        print(top_5)   
        
        if d[0][0] in top_5 or d[1][0] in top_5 or (len(d)<3 or d[2][0] in top_5):
            win+=1
            print("win")
            if d[0][0] in top_5:
                win0+=1
            
            if len(d)<2:
                if d[0][0] in top_5:
                    win1+=1
            elif d[1][0] in top_5:
                win1+=1
                
            if len(d)<2:
                if d[0][0] in top_5:
                    win2+=1
            elif len(d)<3:
                if d[0][0] in top_5 or d[1][0] in top_5:
                    win2+=1
            elif d[2][0] in top_5:
                win2+=1
            
            if len(d)<2:
                if d[0][0] in top_5:
                    win3+=1
            elif d[0][0] in top_5 and d[1][0] in top_5:
                win3+=1
                
            if len(d)<2:
                if d[0][0] in top_5:
                    win4+=1
            elif d[0][0] in top_5 or d[1][0] in top_5:
                win4+=1
             
            if len(d)<2:
                if d[0][0] in top_5:
                    win5+=1
            elif len(d)<3:
                if d[0][0] in top_5 and d[1][0] in top_5:
                    win5+=1
            elif d[0][0] in top_5 and d[1][0] in top_5 and d[2][0] in top_5:
                win5+=1
                
            if len(d)<2:
                if d[0][0] in top_5:
                    win6+=1
            elif len(d)<3:
                if d[0][0] in top_5 or d[1][0] in top_5:
                    win6+=1
            elif d[0][0] in top_5 or d[1][0] in top_5 or d[2][0] in top_5:
                win6+=1
        else:
            lose+=1
        index += step
    
    print(win)
    print(lose)
    print(win/(win+lose))
    print(win0/(win+lose))
    print(win1/(win+lose))
    print(win2/(win+lose))
    print(win3/(win+lose))
    print(win4/(win+lose))
    print(win5/(win+lose))
    print(win6/(win+lose))
    print("Done")
    


