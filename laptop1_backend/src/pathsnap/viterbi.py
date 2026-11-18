"""Deterministic log-space decoding with explicit disconnected segments."""
from collections import Counter
import math
import numpy as np
import pandas as pd
from .hmm import DecoderConfig, STATE_COLUMNS, OBSERVATION_COLUMNS, log_emissions
from .transitions import transition


def decode(observations, states, router, config=None):
    """Return one observation row, selected links, full arc rows and DP diagnostics.

    Only allowlisted label-free columns enter this function. Observation index
    order is preserved; timestamps are never sorted to conceal invalid time.
    """
    config = config or DecoderConfig()
    if list(observations.columns) != OBSERVATION_COLUMNS or list(states.columns) != STATE_COLUMNS:
        raise ValueError('Decoder accepts only the label-free observation/state schema')
    if not observations.index.is_unique or states.duplicated(STATE_COLUMNS[:2]).any():
        raise ValueError('Duplicate observation/candidate identity')
    if not states.gps_observation_index.isin(observations.index).all():
        raise ValueError('Candidate refers to unknown observation')
    if len(states):
        log_emissions(states.positive_probability.to_numpy(), config.probability_floor)
    groups = {i: g.sort_values('candidate_arc_id').reset_index(drop=True)
              for i, g in states.groupby('gps_observation_index', sort=False)}
    output = {}; links = []; route_rows = []; diagnostics = []; layers = []
    segment = -1; previous_obs = None

    def finish():
        nonlocal layers, segment
        if not layers:
            return
        segment += 1
        choice = int(np.argmax(layers[-1]['scores'])); choices = [choice]
        for layer in reversed(layers[1:]):
            choice = int(layer['back'][choice]); choices.append(choice)
        choices.reverse(); route_sequence = []
        for k, (layer, choice) in enumerate(zip(layers, choices)):
            state = layer['states'].iloc[choice]
            aid = int(state.candidate_arc_id); oid = int(layer['oid'])
            output[oid] = dict(gps_observation_index=oid, candidate_arc_id=aid,
                position_along_arc=float(state.position_along_arc), positive_probability=float(state.positive_probability),
                segment_id=segment, status='matched', break_reason=layer['reason'] if k == 0 else '',
                prefix_log_score=float(layer['scores'][choice]))
            if k == 0:
                route_sequence.append((aid, oid, 'selected_start'))
            else:
                previous = layers[k-1]['states'].iloc[choices[k-1]]
                score, route = transition(router, previous, state, layer['dt'], layer['displacement'], config)
                assert math.isfinite(score) and route.feasible
                links.append(dict(from_observation=int(layers[k-1]['oid']), to_observation=oid,
                    from_arc=int(previous.candidate_arc_id), to_arc=aid, segment_id=segment,
                    dt_s=layer['dt'], gps_displacement_m=layer['displacement'], network_distance_m=route.distance_m,
                    transition_log_score=score, intermediate_arcs=len(route.arcs)-2 if len(route.arcs)>1 else 0,
                    route_arcs=list(route.arcs)))
                for j, arc in enumerate(route.arcs[1:]):
                    route_sequence.append((arc, oid, 'selected_end' if j == len(route.arcs)-2 else 'intermediate'))
        for order, (arc, oid, role) in enumerate(route_sequence):
            route_rows.append(dict(segment_id=segment, route_order=order, arc_id=arc,
                to_observation=oid, role=role))
        layers = []

    for oid, obs in observations.iterrows():
        oid = int(oid); reason = ''; dt = math.nan; displacement = float(obs.displacement_m)
        coords_ok = np.isfinite([obs.longitude, obs.latitude, obs.timestamp_s]).all() and abs(obs.longitude)<=180 and abs(obs.latitude)<=90
        group = groups.get(oid, states.iloc[:0]).copy()
        valid = group.candidate_arc_id.isin(router.arcs) & group.position_along_arc.between(0,1)
        group = group.loc[valid].reset_index(drop=True)
        if not coords_ok or group.empty:
            finish(); previous_obs = None
            output[oid] = dict(gps_observation_index=oid, candidate_arc_id=None, position_along_arc=None,
                positive_probability=None, segment_id=None, status='unmatched',
                break_reason='invalid_observation' if not coords_ok else 'no_usable_candidates', prefix_log_score=None)
            continue
        if previous_obs is not None:
            dt = float(obs.timestamp_s - previous_obs.timestamp_s)
            if not math.isfinite(dt) or dt <= 0: reason = 'invalid_timestamp_interval'
            elif dt > config.gap_s: reason = 'temporal_gap'
            elif not math.isfinite(displacement): reason = 'invalid_displacement'
            elif displacement > config.max_speed_mps*dt+config.distance_slack_m: reason = 'gps_jump'
        if reason:
            finish()
        emission = log_emissions(group.positive_probability.to_numpy(), config.probability_floor)
        records = list(group.itertuples(index=False))
        back = None
        if layers:
            left = layers[-1]; matrix = np.full((len(left['states']), len(group)), -np.inf); statuses = Counter()
            for i, a in enumerate(left['records']):
                for j, b in enumerate(records):
                    score, route = transition(router, a, b, dt, displacement, config)
                    matrix[i,j] = score; statuses[route.status] += 1
            totals = left['scores'][:,None] + matrix
            back = np.argmax(totals,axis=0); scores = totals[back,np.arange(len(group))] + emission
            diagnostics.append(dict(from_observation=int(left['oid']), to_observation=oid,
                attempted_pairs=int(matrix.size), feasible_pairs=int(np.isfinite(matrix).sum()),
                feasible_dp_states=int(np.isfinite(scores).sum()), **dict(statuses)))
            if not np.isfinite(scores).any():
                finish(); reason = 'no_feasible_transition'; scores = emission - math.log(len(group)); back = None
        else:
            scores = emission - math.log(len(group))
        layers.append(dict(oid=oid, states=group, scores=scores, back=back,
                           records=records, reason=reason or ('trip_start' if not output else 'after_unmatched'), dt=dt, displacement=displacement))
        previous_obs = obs
    finish()
    selected = pd.DataFrame([output[int(i)] for i in observations.index], columns=[
        'gps_observation_index','candidate_arc_id','position_along_arc','positive_probability','segment_id','status','break_reason','prefix_log_score'])
    selected['candidate_arc_id'] = pd.array(selected.candidate_arc_id,dtype='Int64')
    selected['segment_id'] = pd.array(selected.segment_id,dtype='Int64')
    link_columns=['from_observation','to_observation','from_arc','to_arc','segment_id','dt_s','gps_displacement_m','network_distance_m','transition_log_score','intermediate_arcs','route_arcs']
    return selected, pd.DataFrame(links,columns=link_columns), pd.DataFrame(route_rows,columns=['segment_id','route_order','arc_id','to_observation','role']), pd.DataFrame(diagnostics)
