
# coding: utf-8

import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
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
    file="/home/shaoguang/anaconda3/shaoguang123/bishe_csg/hq2.csv"
    hq=pd.read_csv(file,encoding="utf-8")
    hq1=hq.iloc[:,[0,1,2,4,5]]
    
    #处理数据求得每日各行业板块涨幅
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
    
    data=pd.DataFrame(val,index=Column_list,columns=row_list)
    
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
    
    #PCA降维
    pca=PCA(n_components=10,copy=False)
    A=pca.fit_transform(data)
    
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
    
    
    while index+T < len(A)-step:
        model = hmm.GaussianHMM(n_components= n, covariance_type="spherical", n_iter=1000).fit(A[index:index+T])
        hist_info = [] 
        hiddenStatus = model.predict(A[index:index+T])
        
        #print (hiddenStatus)
        for i in range(index, index+T):
            #hiddenStatu = model.predict(A[index+i : index+i+1])
            score = model.score(A[i: i+1])
            day_tuple = (i, hiddenStatus[i-index], score)
            hist_info.append(day_tuple) 
            
        #print (hist_info)
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
                d[order_list.iloc[pos_diffs[i][0]+1,0]]+=weights[i]/2
            else:
                d[order_list.iloc[pos_diffs[i][0]+1,0]]=weights[i]/2
                    
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




