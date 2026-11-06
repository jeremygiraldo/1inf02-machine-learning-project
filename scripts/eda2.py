import pandas as pd, numpy as np
T=pd.read_csv('/mnt/user-data/attachments/be788647-53c0-5d49-8016-76bc67ff7136/f0f85bfb-Times.csv',dtype={'vehicle_id_id':str})
P=lambda s:s.strip('{}').split(',')
D=pd.DataFrame([[np.nan if x=='-' else float(x) for x in P(s)] for s in T.difference_time])
print('missing per point',D.isna().mean().round(3).tolist())
print('rows complete',D.notna().all(1).mean())
print(D.describe().T[['mean','std','min','50%','max']].round(2))
fin=D.ffill(axis=1).iloc[:,-1]
print('final delay dist',fin.describe(), (fin>2).mean(),(fin<-2).mean())
print('corr diff point 9 vs 17',D[8].corr(D[17]),'pt5 vs 17',D[4].corr(D[17]))
S=[[pd.Timedelta(x) for x in P(s)] for s in T.start_time]
off=pd.Series([tuple(int((t-l[0]).total_seconds()//60) for t in l) for l in S])
print(off.value_counts().head(8))
dur=pd.Series([(l[-1]-l[0]).total_seconds()/60 for l in S]); print('sched duration',dur.value_counts().head())
# actual duration
E=[[pd.Timedelta(x) for x in P(s)] for s in T.end_time]
ad=pd.Series([(e[-1]-e[0]).total_seconds()/60 if e[0]!=pd.Timedelta(0) and e[-1]!=pd.Timedelta(0) else np.nan for e in E])
print('actual dur',ad.describe())
# check difference == round((end-start)/60)
chk=[]
for s,e,d in zip(S,E,D.values):
    for a,b,c in zip(s,e,d):
        if not np.isnan(c): chk.append(((b-a).total_seconds()/60, c))
chk=np.array(chk); print('diff vs computed', np.corrcoef(chk.T)[0,1], np.mean(np.abs(np.trunc(chk[:,0])-chk[:,1])<=0.5), np.mean(np.abs(np.round(chk[:,0])-chk[:,1])<=0.5))
T['hour']=pd.to_datetime(T.time,format='%H:%M:%S').dt.hour
T['fin']=fin; T['wd']=pd.to_datetime(T.date).dt.dayofweek
print(T.groupby('hour').fin.agg(['mean','std','size']).round(2))
print(T.groupby('wd').fin.agg(['mean','size']).round(2))
print(T.groupby('lap').fin.agg(['mean','size']).round(2))
vm=T.groupby('vehicle_id_id').fin.mean(); print('vehicle mean delay spread',vm.describe())
