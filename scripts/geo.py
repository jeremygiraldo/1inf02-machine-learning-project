import pandas as pd, numpy as np
T=pd.read_csv('/mnt/user-data/attachments/be788647-53c0-5d49-8016-76bc67ff7136/f0f85bfb-Times.csv',dtype={'vehicle_id_id':str})
M=pd.read_csv('/mnt/user-data/uploads/Monitoring.csv',dtype={'vehicle_id_id':str})
M['ts']=pd.to_datetime(M.date+' '+M.time); M=M.sort_values('ts')
P=lambda s:s.strip('{}').split(',')
rows=[]
for r in T.itertuples():
    for k,e in enumerate(P(r.end_time)):
        if e!='00:00:00': rows.append((r.vehicle_id_id,pd.Timestamp(r.date+' '+e),k,r.lap))
C=pd.DataFrame(rows,columns=['vehicle_id_id','ts','cp','lap_t']).sort_values('ts')
J=pd.merge_asof(C,M[['ts','vehicle_id_id','latitude','longitude','lap','speed']],on='ts',by='vehicle_id_id',direction='nearest',tolerance=pd.Timedelta('60s'))
print('matched',J.latitude.notna().mean(),'lap agree',(J.lap==J.lap_t).mean())
g=J.groupby('cp')[['latitude','longitude']].agg(['median','std']); print(g.round(5))
g.to_csv('cp.csv')
# per-trip GPS coverage within Monitoring
print('monitoring dates with times',M.date.isin(T.date.unique()).mean())
# speed by hour
M['h']=M.ts.dt.hour; print(M[M.speed>0].groupby('h').speed.mean().round(1).to_dict())
# sample route for plotting
M.sample(60000,random_state=0)[['latitude','longitude','speed']].to_csv('sample.csv',index=False)
