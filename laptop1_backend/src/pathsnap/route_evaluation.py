"""Post-decoding evaluation only; labels never enter the decoder modules."""
import math
import numpy as np
import pandas as pd
from .transitions import transition


def selected_accuracy(selected, labels, alignment):
    """Fixed qualified/reference-present denominator includes unmatched failures."""
    winners=selected.loc[selected.candidate_arc_id.notna(),['gps_observation_index','candidate_arc_id']]
    if winners.gps_observation_index.duplicated().any():raise ValueError('Duplicate selected observation')
    joined=winners.merge(labels[['gps_observation_index','candidate_arc_id','evaluation_target']],
                         on=['gps_observation_index','candidate_arc_id'],how='left',validate='one_to_one')
    correct=int(joined.evaluation_target.eq(1).sum())
    present=int(alignment.reference_candidate_present.sum());qualified=int(alignment.supervised_eligible.sum())
    return dict(raw_observations=len(alignment),qualified_alignments=qualified,candidate_present=present,
        candidate_misses=qualified-present,correct=correct,top1_accuracy=correct/present if present else None,
        top1_including_candidate_misses=correct/qualified if qualified else None,
        correct_per_raw_observation=correct/len(alignment) if len(alignment) else None,
        matched_observations=len(winners))


def diagnose_pairs(selected, observations, router, config):
    chosen=selected.loc[selected.candidate_arc_id.notna()].set_index('gps_observation_index');rows=[]
    ids=list(observations.index)
    for previous,current in zip(ids,ids[1:]):
        dt=float(observations.loc[current,'timestamp_s']-observations.loc[previous,'timestamp_s'])
        displacement=float(observations.loc[current,'displacement_m'])
        row=dict(from_observation=int(previous),to_observation=int(current),dt_s=dt,gps_displacement_m=displacement,
            assessable=False,direct_connected=False,topologically_reachable=False,feasible=False,cross_segment=False,
            network_distance_m=None,category='')
        if not math.isfinite(dt) or dt<=0:row['category']='invalid_timestamp'
        elif dt>config.gap_s:row['category']='temporal_gap'
        elif not math.isfinite(displacement):row['category']='invalid_displacement'
        elif displacement>config.max_speed_mps*dt+config.distance_slack_m:row['category']='gps_jump'
        elif previous not in chosen.index or current not in chosen.index:row['category']='missing_prediction'
        else:
            a,b=chosen.loc[previous],chosen.loc[current];aid,bid=int(a.candidate_arc_id),int(b.candidate_arc_id)
            forward=aid==bid and b.position_along_arc>=a.position_along_arc
            direct=forward or (aid!=bid and router.arcs[aid][1]==router.arcs[bid][0])
            reachable=forward or router.reachable(router.arcs[aid][1],router.arcs[bid][0])
            score,route=transition(router,a,b,dt,displacement,config)
            row.update(assessable=True,direct_connected=bool(direct),topologically_reachable=bool(reachable),
                feasible=math.isfinite(score),network_distance_m=route.distance_m if math.isfinite(route.distance_m) else None,
                category=route.status,cross_segment=bool(a.segment_id!=b.segment_id) if 'segment_id' in chosen else False)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_pairs(frame):
    eligible=frame.loc[frame.assessable];n=len(eligible)
    return dict(adjacent_pairs=len(frame),assessable_pairs=n,direct_connected=int(eligible.direct_connected.sum()),
        direct_continuity=float(eligible.direct_connected.mean()) if n else None,
        topologically_unreachable=int((~eligible.topologically_reachable).sum()),
        feasible_pairs=int(eligible.feasible.sum()),infeasible_pairs=int((~eligible.feasible).sum()),
        bounded_feasibility=float(eligible.feasible.mean()) if n else None,
        cross_segment_pairs=int(eligible.cross_segment.sum()),
        within_segment_infeasible=int((~eligible.feasible & ~eligible.cross_segment).sum()),
        network_distance_sum_m=float(eligible.network_distance_m.sum()))
