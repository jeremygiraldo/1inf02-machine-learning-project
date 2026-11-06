import pandas as pd, numpy as np
T=pd.read_csv('/mnt/user-data/attachments/be788647-53c0-5d49-8016-76bc67ff7136/f0f85bfb-Times.csv',dtype={'vehicle_id_id':str})
P=lambda s:s.strip('{}').split(',')
D=pd.DataFrame([[np.nan if x=='-' else float(x) for x in P(s)] for s in T.difference_time])
E=pd.DataFrame([[np.nan if x=='00:00:00' else pd.Timedelta(x).total_seconds()/60 for x in P(s)] for s in T.end_time])
S=pd.DataFrame([[pd.Timedelta(x).total_seconds()/60 for x in P(s)] for s in T.start_time])
mono=(S.diff(axis=1).iloc[:,1:]>=0).all(axis=1)
print('non-monotonic schedules',(~mono).sum())
bad=D.abs()>60; print('|diff|>60 cells',bad.sum().sum(), 'rows',bad.any(axis=1).sum())
D=D.mask(bad)
D=D[mono]; T2=T[mono].copy()
print(D.describe().T[['count','mean','std','min','50%','max']].round(2))
y=D[17]; print('final delay clean',y.describe().round(2)); print('late>3',(y>3).mean(),'early<-1',(y<-1).mean())
# class bins
c=pd.cut(y,[-99,-1,3,99],labels=['adelantado','puntual','retrasado']); print(c.value_counts(normalize=True))
T2['hour']=S[mono][0]//60; T2['wd']=pd.to_datetime(T2.date).dt.dayofweek; T2['sched']=(S[mono][17]-S[mono][0]).values
X=pd.concat([D.iloc[:,:10].add_prefix('d'),T2[['hour','wd','lap','sched']].reset_index(drop=True).set_index(D.index),T2.vehicle_id_id.astype(int).rename('veh')],axis=1)
m=y.notna(); X=X[m]; yy=y[m]; dates=T2.date[m]
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE
tr=dates<'2025-01-25'; te=~tr
print('train',tr.sum(),'test',te.sum())
pers=X.loc[te,[f'd{i}' for i in range(10)]].ffill(axis=1).iloc[:,-1].fillna(0)
print('MAE zero',MAE(yy[te],np.zeros(te.sum())),'persistence',MAE(yy[te],pers))
g=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05).fit(X[tr],yy[tr]); print('HGB',MAE(yy[te],g.predict(X[te])))
Xf=X.fillna(X[tr].median())
rf=RandomForestRegressor(300,min_samples_leaf=5,n_jobs=-1,random_state=0).fit(Xf[tr],yy[tr]); print('RF',MAE(yy[te],rf.predict(Xf[te])))
r=Ridge().fit(Xf[tr],yy[tr]); print('Ridge',MAE(yy[te],r.predict(Xf[te])))
print(pd.Series(rf.feature_importances_,X.columns).sort_values(ascending=False).head(8).round(3))
# trip-time target
tt=(E[mono][17]-E[mono][0]); tt=tt.where(tt>0); print('actual trip time',tt.describe().round(1))
