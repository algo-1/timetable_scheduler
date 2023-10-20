from collections import namedtuple

# TODO - design better abstraction as it seems the do not difer much in terms of field_names (see - http://jeffreykingston.id.au/cgi-bin/hseval.cgi?op=spec&part=constraints), just the deviation?? where does this come in, eval? so we have an enum that specifies type and a general constraint??

AssignTimeConstraint = namedtuple("AssignTimeConstraint", [])

SplitEventsConstraint = namedtuple("SplitEventsConstraint", [])

DistributeSplitEventsConstraint = namedtuple("DistributeSplitEventsConstraint", [])

PreferResourcesConstraint = namedtuple("PreferResourcesConstraint", [])

PreferTimesConstraint = namedtuple("PreferTimesConstraint", [])

AvoidSplitAssignmentsConstraint = namedtuple("AvoidSplitAssignmentsConstraint", [])

SpreadEventsConstraint = namedtuple("SpreadEventsConstraint", [])

LinkEventsConstraint = namedtuple("LinkEventsConstraint", [])

OrderEventsConstraint = namedtuple("OrderEventsConstraint", [])

AvoidClashesConstraint = namedtuple("AvoidClashesConstraint", [])

AvoidUnavailableTimesConstraint = namedtuple("AvoidUnavailableTimesConstraint", [])

LimitIdleTimesConstraint = namedtuple("LimitIdleTimesConstraint", [])

ClusterBusyTimesConstraint = namedtuple("ClusterBusyTimesConstraint", [])

LimitBusyTimesConstraint = namedtuple("LimitBusyTimesConstraint", [])

LimitWorkloadConstraint = namedtuple("LimitWorkloadConstraint", [])
