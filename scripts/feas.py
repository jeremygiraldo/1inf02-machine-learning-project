import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score, f1_score
T=pd.read_csv('/mnt/user-data/attachments/be788647-53c0-5d49-8016-76bc67ff7136/f0f85bfb-Times.csv',dtype={'vehicle_id_id':str})
P=lambda s:s.strip('{}').split(',')
D=pd.DataFrame([[np.nan if x=='-' else float(x) for x in P(s)] for s in T.difference_time])
S=pd.DataFrame([[pd.Timedelta(x).total_seconds()/60 for x in P(s)] for s in T.start_time])
mono=(S.diff(axis=1).iloc[:,1:]>=0).all(axis=1); D=D.mask(D.abs()>60)[mono]; T=T[mono]; S=S[mono]
X=D.iloc[:,:10].add_prefix('d').copy()
X['hour']=S[0]//60; X['wd']=pd.to_datetime(T.date).dt.dayofweek; X['lap']=T.lap; X['sched']=S[17]-S[0]; X['veh']=T.vehicle_id_id.astype(int)
y=D[17]; m=y.notna(); X,y,dt=X[m],y[m],T.date[m]
tr=dt<='2025-01-21'; va=(dt>'2025-01-21')&(dt<='2025-01-26'); te=dt>='2025-01-27'
print(tr.sum(),va.sum(),te.sum())
trv=tr|va
def rep(n,p): e=np.abs(y[te]-p); print(f'{n:12s} MAE {e.mean():.2f} RMSE {np.sqrt((e**2).mean()):.2f} R2 {r2_score(y[te],p):.2f} <=2 {np.mean(e<=2)*100:.0f}%')
rep('cero',np.zeros(te.sum()))
rep('persist',X.loc[te,[f'd{i}' for i in range(10)]].ffill(axis=1).iloc[:,-1].fillna(0))
hm=y[trv].groupby(X.hour[trv]).mean(); rep('hist hora',X.hour[te].map(hm).fillna(y[trv].mean()))
Xf=X.fillna(X[trv].median())
rep('Ridge',Ridge().fit(Xf[trv],y[trv]).predict(Xf[te]))
rep('RF',RandomForestRegressor(300,min_samples_leaf=5,n_jobs=-1,random_state=0).fit(Xf[trv],y[trv]).predict(Xf[te]))
rep('HGB',HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=0).fit(X[trv],y[trv]).predict(X[te]))
c=pd.cut(y,[-99,-1.5,3,99],labels=[0,1,2]).astype(int)
pc=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,random_state=0).fit(X[trv],c[trv]).predict(X[te])
print('clf F1 macro',f1_score(c[te],pc,average='macro'),'dist',c.value_counts(normalize=True).round(3).to_dict())
