"""Log transition potentials: directed feasibility and displacement agreement."""
import math
from .routing import Route


def transition(router, left, right, dt_s, displacement_m, config):
    if not math.isfinite(dt_s) or dt_s <= 0 or dt_s > config.gap_s:
        return -math.inf, Route(math.inf, (), 'temporal_boundary')
    if not math.isfinite(displacement_m) or displacement_m < 0:
        return -math.inf, Route(math.inf, (), 'invalid_displacement')
    route = router.route(int(left.candidate_arc_id), float(left.position_along_arc),
                         int(right.candidate_arc_id), float(right.position_along_arc))
    if not route.feasible:
        return -math.inf, route
    if route.distance_m > config.max_speed_mps * dt_s + config.distance_slack_m:
        return -math.inf, Route(route.distance_m, (), 'implausible_speed')
    return -abs(route.distance_m - displacement_m) / config.beta_m, route
