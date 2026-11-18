"""Non-destructive quality flags and causal temporal measurements."""
import numpy as np
import pandas as pd
from .projection import GEOD
from .quality import valid_coordinates

def clean_observations(track):
    out=track.copy()
    out['invalid_coordinates']=~valid_coordinates(track)
    out['duplicate_record']=track.duplicated()
    t=track.timestamp_s.to_numpy(); dt=np.r_[np.nan,np.diff(t)]
    out['dt_previous_s']=dt
    out['bad_timestamp']=~np.isfinite(t) | (np.isfinite(dt)&(dt<=0))
    out['large_gap']=dt>5
    valid=~out.invalid_coordinates.to_numpy()
    distance=np.full(len(track),np.nan); heading=distance.copy()
    ids=np.flatnonzero(valid[1:]&valid[:-1])+1
    if len(ids):
        az,_,d=GEOD.inv(track.longitude.to_numpy()[ids-1],track.latitude.to_numpy()[ids-1],
                       track.longitude.to_numpy()[ids],track.latitude.to_numpy()[ids])
        distance[ids]=d; heading[ids]=az%360
    speed=np.full(len(track),np.nan)
    np.divide(distance*3.6,dt,out=speed,where=np.isfinite(dt)&(dt>0))
    out['displacement_m']=distance; out['speed_kmh']=speed
    out['speed_outlier']=speed>180; out['gps_jump']=distance>500
    boundary=out.large_gap|out.bad_timestamp|out.invalid_coordinates|out.invalid_coordinates.shift(fill_value=False)|out.speed_outlier|out.gps_jump
    if len(boundary): boundary.iloc[0]=True
    out['temporal_segment']=boundary.cumsum().astype('int64')-1
    usable_heading=(distance>=5)&~boundary.to_numpy()&np.isfinite(speed)
    heading[~usable_heading]=np.nan
    out['gps_heading_deg']=heading
    out['supervision_quality_ok']=~(out.invalid_coordinates|out.bad_timestamp|out.duplicate_record|out.speed_outlier|out.gps_jump)
    return out
