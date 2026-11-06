import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T=pd.read_csv('/mnt/user-data/attachments/be788647-53c0-5d49-8016-76bc67ff7136/f0f85bfb-Times.csv',dtype={'vehicle_id_id':str})
P=lambda s:s.strip('{}').split(',')
D=pd.DataFrame([[np.nan if x=='-' else float(x) for x in P(s)] for s in T.difference_time])
S=pd.DataFrame([[pd.Timedelta(x).total_seconds()/60 for x in P(s)] for s in T.start_time])
mono=(S.diff(axis=1).iloc[:,1:]>=0).all(axis=1); D=D.mask(D.abs()>60)[mono]
q=D.quantile([.25,.5,.75]).T; q['mean']=D.mean(); print(q.round(1).to_string())
print('incomplete rows (any NaN) after clean',D.isna().any(axis=1).mean())
# kmeans on complete profiles
from sklearn.cluster import KMeans; from sklearn.metrics import silhouette_score
C=D.dropna(); print('complete profiles',len(C))
for k in [2,3,4,5,6]:
    km=KMeans(k,n_init=10,random_state=0).fit(C); print(k,round(silhouette_score(C,km.labels_),3))
km=KMeans(3,n_init=10,random_state=0).fit(C); print(pd.DataFrame(km.cluster_centers_).round(1).iloc[:,[0,5,9,13,17]], np.bincount(km.labels_))
cp=pd.read_csv('cp.csv',header=[0,1],index_col=0)
s=pd.read_csv('sample.csv'); s=s[(s.latitude.between(-13.57,-13.50))&(s.longitude.between(-72.01,-71.84))]
fig,ax=plt.subplots(figsize=(9,4.2),dpi=200)
ax.scatter(s.longitude,s.latitude,s=0.3,c='#9aa5b1',alpha=.25,lw=0)
la=cp[('latitude','median')]; lo=cp[('longitude','median')]
ax.scatter(lo,la,s=40,c='#c2410c',zorder=3,edgecolor='white',lw=.8)
off={5:(6,6),6:(6,-10),7:(-16,4),10:(6,-4),11:(6,4),12:(-14,6),13:(4,6),14:(4,-11),0:(4,4),1:(-14,-10),2:(4,6),15:(-6,-12),16:(4,-12),17:(-16,4),8:(6,-6),9:(4,-10),3:(4,4),4:(4,4)}
for i,(x,y) in enumerate(zip(lo,la)): ax.annotate(f'P{i+1}',(x,y),xytext=off.get(i,(4,4)),textcoords='offset points',fontsize=7,color='#1f2937')
ax.set_aspect(1/np.cos(np.deg2rad(13.54))); ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
ax.set_title('Ruta observada (muestra de 60 000 puntos GPS) y 18 puntos de control',fontsize=9,loc='left')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('ruta.png')
q.round(2).to_csv('delay_by_cp.csv')
