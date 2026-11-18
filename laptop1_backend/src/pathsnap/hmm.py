"""Frozen HMM scoring configuration and strictly label-free emission inputs."""
from dataclasses import dataclass, asdict
import numpy as np
from .features import FEATURES, IDENTIFIERS

STATE_COLUMNS = ['gps_observation_index', 'candidate_arc_id', 'position_along_arc', 'positive_probability']
OBSERVATION_COLUMNS = ['longitude', 'latitude', 'timestamp_s', 'displacement_m']


@dataclass(frozen=True)
class DecoderConfig:
    gap_s: float = 5.
    beta_m: float = 50.
    max_speed_mps: float = 50.
    distance_slack_m: float = 75.
    probability_floor: float = 1e-12
    max_settled_nodes: int = 4000
    cache_sources: int = 1024

    def __post_init__(self):
        values = [self.gap_s, self.beta_m, self.max_speed_mps, self.distance_slack_m]
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Invalid HMM scale or movement bound')
        if not 0 < self.probability_floor < .5 or self.max_settled_nodes < 1 or self.cache_sources < 1:
            raise ValueError('Invalid numerical or routing budget')

    @property
    def routing_cutoff_m(self):
        return self.max_speed_mps * self.gap_s + self.distance_slack_m

    def to_dict(self):
        return asdict(self)


def log_emissions(probabilities, floor=1e-12):
    """Normalize clipped binary scores over the available states, then log.

    These are discriminative state potentials, not calibrated GPS likelihoods.
    Normalization is observation-constant and preserves relative candidate odds.
    """
    p = np.asarray(probabilities, dtype=float)
    if not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Require finite probabilities in [0,1]')
    p = np.clip(p, floor, 1 - floor)
    return np.log(p) - np.log(p.sum())


def emission_states(features, probabilities):
    if list(features.columns) != IDENTIFIERS + FEATURES:
        raise ValueError('Only the exact inference feature schema is accepted')
    if list(probabilities.columns) != IDENTIFIERS + ['positive_probability']:
        raise ValueError('Probability input must not contain references or labels')
    if features.duplicated(IDENTIFIERS).any() or probabilities.duplicated(IDENTIFIERS).any():
        raise ValueError('Duplicate candidate identity')
    joined = features.merge(probabilities, on=IDENTIFIERS, how='outer', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all() or joined.trip_id.nunique() > 1:
        raise ValueError('Require identical candidate identities for one trip')
    return joined[STATE_COLUMNS]
