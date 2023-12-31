from __future__ import annotations
from collections import defaultdict, namedtuple
from inspect import isclass
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod


from XHSTTS.utils import (
    Cost,
    Cost_Function_Type,
    cost,
    cost_function_to_enum,
)


class Constraint(ABC):
    def __init__(
        self,
        XMLConstraint: ET.Element,
        instance_resource_groups,
        instance_event_groups,
        instance_time_groups,
        instance_events,
    ):
        self.name: str = XMLConstraint.find("Name").text
        self.required: bool = XMLConstraint.find("Required").text == "true"
        self.weight = int(XMLConstraint.find("Weight").text)
        self.cost_function: Cost_Function_Type = cost_function_to_enum(
            XMLConstraint.find("CostFunction").text
        )
        self.resource_references = []
        self.instance_resource_groups = instance_resource_groups
        self.events = []
        self.instance_event_groups = instance_event_groups
        self.instance_events = instance_events
        self.instance_time_groups = instance_time_groups
        self._parse_applies_to(XMLConstraint.find("AppliesTo"))

    @abstractmethod
    def evaluate(self, solution: list[XHSTTSInstance.SolutionEvent]) -> int:
        pass

    def is_required(self):
        return self.required

    def get_name(self):
        return self.name

    def get_cost_function(self):
        return self.get_cost_function

    def get_weight(self):
        return self.weight

    def _parse_applies_to(self, XMLAppliesTo: ET.Element):
        XMLResourceGroups = XMLAppliesTo.find("ResourceGroups")
        if XMLResourceGroups:
            for XMLResourceGroup in XMLResourceGroups.findall("ResourceGroup"):
                group_reference = XMLResourceGroup.attrib["Reference"]
                self.resource_references.extend(
                    list(
                        map(
                            lambda x: x.Reference,
                            self.instance_resource_groups[group_reference],
                        )
                    )
                )

        XMLResources = XMLAppliesTo.find("Resources")
        if XMLResources:
            for XMLResource in XMLResources:
                self.resource_references.append(XMLResource.attrib["Reference"])

        XMLEventGroups = XMLAppliesTo.find("EventGroups")
        if XMLEventGroups:
            for XMLEventGroup in XMLEventGroups.findall("EventGroup"):
                self.events.extend(
                    self.instance_event_groups[XMLEventGroup.attrib["Reference"]]
                )

        XMLEvents = XMLAppliesTo.find("Events")
        if XMLEvents:
            for XMLEvent in XMLEvents:
                self.events.append(self.instance_events[XMLEvent.attrib["Reference"]])

        XMLEventPairs = XMLAppliesTo.find("EventPairs")
        if XMLEventPairs:
            raise Exception("Not implemented yet")  # TODO

    def hasResource(
        self,
        resource_reference,
        solution_event_resources: list[XHSTTSInstance.SolutionEventResource],
    ):
        for reference, _ in solution_event_resources:
            if reference == resource_reference:
                return True
        return False


class AssignTimeConstraint(Constraint):
    def __init__(self, *args):
        super().__init__(*args)

    def evaluate(self, solution):
        deviation = 0
        for event in self.events:
            if not event.PreAssignedTimeReference:
                seen = False
                for solution_event in solution:
                    if solution_event.InstanceEventReference == event.Reference:
                        seen = True
                        if not solution_event.TimeReference:
                            deviation += solution_event.Duration
                if not seen:
                    deviation += event.Duration

        return cost(deviation, self.weight, self.cost_function)


class AssignResourceConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.role = XMLConstraint.find("Role").text

    def is_assigned(
        self, solution_event_resources: list[XHSTTSInstance.SolutionEventResource]
    ):
        for _, role in solution_event_resources:
            if role == self.role:
                return True
        return False

    def evaluate(self, solution):
        deviation = 0
        for event in self.events:
            for resource in event.Resources:
                if not resource.Reference and resource.Role == self.role:
                    for solution_event in solution:
                        deviation += (
                            solution_event.Duration
                            if solution_event.InstanceEventReference == event.Reference
                            and (not self.is_assigned(solution_event.Resources))
                            else 0
                        )

        return cost(deviation, self.weight, self.cost_function)


class PreferResourcesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.role = XMLConstraint.find("Role").text
        self.preferred_resources = set()
        self._parse_preferred_resources(XMLConstraint)

    def is_assigned_and_not_preferred(
        self, solution_event_resources: list[XHSTTSInstance.SolutionEventResource]
    ):
        for ref, role in solution_event_resources:
            if ref and role == self.role:
                if ref not in self.preferred_resources:
                    return True
        return False

    def evaluate(self, solution):
        deviation = 0
        for event in self.events:
            for resource in event.Resources:
                if not resource.Reference and resource.Role == self.role:
                    for solution_event in solution:
                        deviation += (
                            solution_event.Duration
                            if solution_event.InstanceEventReference == event.Reference
                            and self.is_assigned_and_not_preferred(
                                solution_event.Resources
                            )
                            else 0
                        )

        return cost(deviation, self.weight, self.cost_function)

    def _parse_preferred_resources(self, XMLConstraint: ET.Element):
        XMLResourceGroups = XMLConstraint.find("ResourceGroups")
        if XMLResourceGroups:
            for XMLResourceGroup in XMLResourceGroups.findall("ResourceGroup"):
                resource_group_ref = XMLResourceGroup.attrib["Reference"]
                for resource in self.instance_resource_groups[resource_group_ref]:
                    self.preferred_resources.add(resource.Reference)

        XMLResources = XMLConstraint.find("Resources")
        if XMLResources:
            for XMLResource in XMLResources.findall("Resource"):
                self.preferred_resources.add(XMLResource.attrib["Reference"])


class AvoidClashesConstraint(Constraint):
    def __init__(self, *args):
        super().__init__(*args)

    def evaluate(self, solution):
        deviation = 0
        for resource_ref in self.resource_references:
            times = set()
            for solution_event in solution:
                if self.hasResource(resource_ref, solution_event.Resources):
                    if solution_event.TimeReference in times:
                        deviation += 1
                    else:
                        times.add(solution_event.TimeReference)

        return cost(deviation, self.weight, self.cost_function)


class SplitEventsConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.min_duration = int(XMLConstraint.find("MinimumDuration").text)
        self.max_duration = int(XMLConstraint.find("MaximumDuration").text)
        self.min_amount = int(XMLConstraint.find("MinimumAmount").text)
        self.max_amount = int(XMLConstraint.find("MaximumAmount").text)

    def evaluate(self, solution):
        deviation = 0
        for event in self.events:
            violates_duration_count = 0
            amount_count = 0
            for solution_event in solution:
                if solution_event.InstanceEventReference == event.Reference:
                    amount_count += 1
                    if (
                        solution_event.Duration < self.min_duration
                        or solution_event.Duration > self.max_duration
                    ):
                        violates_duration_count += 1
            deviation += violates_duration_count
            if amount_count < self.min_amount:
                deviation += self.min_amount - amount_count
            elif amount_count > self.max_amount:
                deviation += amount_count - self.max_amount

        return cost(deviation, self.weight, self.cost_function)


class DistributeSplitEventsConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.duration = int(XMLConstraint.find("Duration").text)
        self.min = int(XMLConstraint.find("Minimum").text)
        self.max = int(XMLConstraint.find("Maximum").text)

    def evaluate(self, solution):
        deviation = 0
        for event in self.events:
            count = 0
            for solution_event in solution:
                if solution_event.InstanceEventReference == event.Reference:
                    if solution_event.Duration == self.duration:
                        count += 1
            if count < self.min:
                deviation += self.min - count
            elif count > self.max:
                deviation += count - self.max

        return cost(deviation, self.weight, self.cost_function)


class PreferTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        XMLDuration = XMLConstraint.find("Duration")
        self.duration = int(XMLDuration.text) if XMLDuration else None
        self.preferred_times = set()
        self._parse_preferred_times(XMLConstraint)

    def evaluate(self, solution):
        deviation = 0
        for event in self.events:
            for solution_event in solution:
                if solution_event.InstanceEventReference == event.Reference:
                    if (
                        solution_event.TimeReference
                        and solution_event.TimeReference not in self.preferred_times
                    ):
                        if self.duration:
                            if self.duration == solution_event.Duration:
                                deviation += solution_event.Duration
                        else:
                            deviation += solution_event.Duration

        return cost(deviation, self.weight, self.cost_function)

    def _parse_preferred_times(self, XMLConstraint: ET.Element):
        XMLTimeGroups = XMLConstraint.find("TimeGroups")
        if XMLTimeGroups:
            for XMLTimeGroup in XMLTimeGroups.findall("TimeGroup"):
                ref = XMLTimeGroup.attrib["Reference"]
                for time in self.instance_time_groups[ref]:
                    self.preferred_times.add(time.Reference)

        XMLTimes = XMLConstraint.find("Times")
        if XMLTimes:
            for XMLTime in XMLTimes.findall("Time"):
                self.preferred_times.add(XMLTime.attrib["Reference"])


class SpreadEventsConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.time_groups = []
        self._parse_time_groups(XMLConstraint)

    def evaluate(self, solution):
        deviation = 0
        for event_group in self.instance_event_groups:
            event_group_events = set(
                [e.Reference for e in self.instance_event_groups[event_group]]
            )
            for refs, minimum, maximum in self.time_groups:
                count = 0
                for solution_event in solution:
                    if solution_event.InstanceEventReference in event_group_events:
                        if (
                            solution_event.TimeReference
                            and solution_event.TimeReference in refs
                        ):
                            count += 1
                if count < minimum:
                    deviation += minimum - count
                elif count > maximum:
                    deviation += count - maximum

        return cost(deviation, self.weight, self.cost_function)

    def _parse_time_groups(self, XMLConstraint: ET.Element):
        XMLTimeGroups = XMLConstraint.find("TimeGroups")
        if XMLTimeGroups:
            for XMLTimeGroup in XMLTimeGroups.findall("TimeGroup"):
                ref = XMLTimeGroup.attrib["Reference"]
                minimum = int(XMLTimeGroup.find("Minimum").text)
                maximum = int(XMLTimeGroup.find("Maximum").text)
                self.time_groups.append(
                    (
                        set([t.Reference for t in self.instance_time_groups[ref]]),
                        minimum,
                        maximum,
                    )
                )


class AvoidUnavailableTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.time_refs = set()
        self._parse_time_refs(XMLConstraint)

    def evaluate(self, solution):
        deviation = 0
        for resource_ref in self.resource_references:
            for time_ref in self.time_refs:
                found = False
                for solution_event in solution:
                    if (
                        solution_event.TimeReference
                        and solution_event.TimeReference == time_ref
                        and self.hasResource(resource_ref, solution_event.Resources)
                    ):
                        found = True
                        break
                if found:
                    deviation += 1

        return cost(deviation, self.weight, self.cost_function)

    def _parse_time_refs(self, XMLConstraint: ET.Element):
        XMLTimeGroups = XMLConstraint.find("TimeGroups")
        if XMLTimeGroups:
            for XMLTimeGroup in XMLTimeGroups.findall("TimeGroup"):
                ref = XMLTimeGroup.attrib["Reference"]
                for time in self.instance_time_groups[ref]:
                    self.time_refs.add(time.Reference)

        XMLTimes = XMLConstraint.find("Times")
        if XMLTimes:
            for XMLTime in XMLTimes.findall("Time"):
                self.time_refs.add(XMLTime.attrib["Reference"])


class LimitIdleTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.min = int(XMLConstraint.find("Minimum").text)
        self.max = int(XMLConstraint.find("Maximum").text)
        self.time_groups = []
        self._parse_time_groups(XMLConstraint)

    def get_resource_status(
        self,
        resource_ref,
        timegroup: tuple,
        solution: list[XHSTTSInstance.SolutionEvent],
    ):
        return [
            True
            if any(
                sol_event.TimeReference
                and time_ref == sol_event.TimeReference
                and self.hasResource(resource_ref, sol_event.Resources)
                for sol_event in solution
            )
            else False
            for time_ref in timegroup
        ]

    def get_idle_times_count(self, resource_status: list[bool]):
        first_true = False
        count = 0
        res = 0
        for value in resource_status:
            if value:
                if not first_true:
                    first_true = True
                else:
                    # add count to result and reset count
                    res += count
                    count = 0
            else:
                if first_true:
                    count += 1
        return res

    def evaluate(self, solution):
        deviation = 0
        for resource_ref in self.resource_references:
            idle_times_count = 0
            for timegroup in self.time_groups:
                resource_status = self.get_resource_status(
                    resource_ref, timegroup, solution
                )
                idle_times_count += self.get_idle_times_count(resource_status)
            if idle_times_count < self.min:
                deviation += self.min - idle_times_count
            elif idle_times_count > self.max:
                deviation += idle_times_count - self.max

        return cost(deviation, self.weight, self.cost_function)

    def _parse_time_groups(self, XMLConstraint: ET.Element):
        XMLTimeGroups = XMLConstraint.find("TimeGroups")
        if XMLTimeGroups:
            for XMLTimeGroup in XMLTimeGroups.findall("TimeGroup"):
                ref = XMLTimeGroup.attrib["Reference"]
                self.time_groups.append(
                    tuple(t.Reference for t in self.instance_time_groups[ref])
                )


class ClusterBusyTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element, *args):
        super().__init__(XMLConstraint, *args)
        self.min = int(XMLConstraint.find("Minimum").text)
        self.max = int(XMLConstraint.find("Maximum").text)
        self.time_groups: list[set] = []
        self._parse_time_groups(XMLConstraint)

    def evaluate(self, solution):
        deviation = 0
        for resource_ref in self.resource_references:
            count = 0
            for timegroup in self.time_groups:
                found = False
                for sol_event in solution:
                    if sol_event.TimeReference:
                        if sol_event.TimeReference in timegroup:
                            if self.hasResource(resource_ref, sol_event.Resources):
                                found = True
                                break
                if found:
                    count += 1

            if count < self.min:
                deviation += self.min - count
            elif count > self.max:
                deviation += count - self.max

        return cost(deviation, self.weight, self.cost_function)

    def _parse_time_groups(self, XMLConstraint: ET.Element):
        XMLTimeGroups = XMLConstraint.find("TimeGroups")
        if XMLTimeGroups:
            for XMLTimeGroup in XMLTimeGroups.findall("TimeGroup"):
                ref = XMLTimeGroup.attrib["Reference"]
                self.time_groups.append(
                    set([t.Reference for t in self.instance_time_groups[ref]])
                )


class XHSTTSInstance:
    # TODO - remove redundant constants & convert named tuples to dataclasses
    TIME_GROUP_NAMES = ["TimeGroup", "Day", "Week"]
    EVENT_GROUP_NAMES = ["EventGroup", "Course"]
    CONSTRAINT_NAMES = [
        "AssignResourceConstraint",
        "AssignTimeConstraint",
        "PreferResourcesConstraint",
        "AvoidClashesConstraint",
    ]
    Time = namedtuple("Time", ["Reference", "Name", "TimeGroupReferences"])
    TimeGroup = namedtuple("TimeGroup", ["Name", "Type"])
    EventGroup = namedtuple("EventGroup", ["Name", "Type"])
    ResourceGroup = namedtuple("ResourceGroup", ["Name", "ResourceTypeReference"])
    Resource = namedtuple(
        "Resource",
        ["Reference", "Name", "ResourceTypeReference", "ResourceGroupReferences"],
    )
    Event = namedtuple(
        "Event",
        [
            "Reference",
            "Name",
            "Duration",
            "Workload",
            "PreAssignedTimeReference",
            "Resources",  # List of EventResource
            "ResourceGroupReferences",
            "CourseReference",
            "EventGroupReferences",
        ],
    )
    EventResource = namedtuple(
        "EventResource", ["Reference", "Role", "ResourceTypeReference", "Workload"]
    )
    SolutionEvent = namedtuple(
        "SolutionEvent",
        [
            "InstanceEventReference",
            "Duration",
            "TimeReference",
            "Resources",  # List of SolutionEventResource
        ],
    )
    SolutionEventResource = namedtuple("SolutionEventResource", ["Reference", "Role"])

    def __init__(self, XMLInstance: ET.Element, XMLSolutions: list[ET.Element]):
        """
        XMLInstance : The XHSTTS XML representation of the Instance.\n
        XMLSolutions : List of all solutions in the dataset that reference the XMLInstance.

        """
        self.TimeGroups = {}
        self.Times = {}
        self.ResourceTypes = {}
        self.ResourceGroups = {}
        self.Resources = {}
        self.Events = {}
        self.EventGroups = {}
        self.instance_event_groups = defaultdict(
            list
        )  # {group_id : list of events that reference this group}
        self.instance_resource_groups = defaultdict(
            list
        )  # {group_id : list of resources that reference this group}
        self.instance_time_groups = defaultdict(
            list
        )  # {group_id : list of times that are in this time group (timegroup can be Day or Week)) }
        self.Constraints: list[Constraint] = []
        self.Solutions = []
        self._parse_times(XMLInstance.find("Times"))
        self._parse_resources(XMLInstance.find("Resources"))
        self._parse_events(XMLInstance.find("Events"))
        self._parse_constraints(
            XMLInstance.find("Constraints")
        )  # must be parsed after the above due to references
        self._parse_solutions(
            XMLSolutions
        )  # must be parsed after Events because of duration fallback!

    def get_events(self):
        return self.Events

    def get_times(self):
        return self.Times

    def get_resources(self):
        return self.Resources

    def get_constraints(self):
        return self.Constraints

    def get_solutions(self):
        return self.Solutions

    def _parse_times(self, XMLTimes: ET.Element):
        XMLTimesGroups = XMLTimes.find("TimeGroups")
        if XMLTimesGroups:
            self.TimeGroups = self._parse_TimeGroups(XMLTimesGroups)

        XMLTimeList = XMLTimes.findall("Time")
        for XMLTime in XMLTimeList:
            time_Id = XMLTime.attrib["Id"]
            Name = XMLTime.find("Name").text
            TimeGroup_references = [
                element.attrib["Reference"]
                for element in (
                    XMLTime.findall(".//Day")
                    + XMLTime.findall(".//Week")
                    + XMLTime.findall(".//TimeGroup")
                )
            ]
            self.Times[time_Id] = XHSTTSInstance.Time(
                Reference=time_Id, Name=Name, TimeGroupReferences=TimeGroup_references
            )

        # populate instance_time_groups
        # NOTE: Assumes that self.Times dictionary is ordered as the order of times as specified in the xml matter. Python 3.11 is used for development/ Python guarantees this behaviour in versions 3.6+
        for _, time in self.Times.items():
            for ref in time.TimeGroupReferences:
                self.instance_time_groups[ref].append(time)

    def _parse_TimeGroups(self, XMLTimeGroups: ET.Element):
        return {
            element.attrib["Id"]: XHSTTSInstance.TimeGroup(
                Name=element.find("Name").text, Type=element.tag
            )
            for element in XMLTimeGroups
        }

    def _parse_resources(self, XMLResources: ET.Element):
        XMLResourceTypes = XMLResources.find("ResourceTypes")
        self.ResourceTypes = {
            XMLResourceType.attrib["Id"]: XMLResourceType.find("Name").text
            for XMLResourceType in XMLResourceTypes.findall("ResourceType")
        }

        XMLResourceGroups = XMLResources.find("ResourceGroups")
        self.ResourceGroups = {
            XMLResourceGroup.attrib["Id"]: XHSTTSInstance.ResourceGroup(
                Name=XMLResourceGroup.find("Name").text,
                ResourceTypeReference=XMLResourceGroup.find("ResourceType").attrib[
                    "Reference"
                ],
            )
            for XMLResourceGroup in XMLResourceGroups.findall("ResourceGroup")
        }

        self.Resources = {
            XMLResource.attrib["Id"]: XHSTTSInstance.Resource(
                XMLResource.attrib["Id"],
                XMLResource.find("Name").text,
                XMLResource.find("ResourceType").attrib["Reference"],
                list(
                    map(
                        lambda x: x.attrib["Reference"],
                        XMLResource.find("ResourceGroups").findall("ResourceGroup"),
                    )
                ),
            )
            for XMLResource in XMLResources.findall("Resource")
        }

        # populate instance_resource_groups
        for _, resource in self.Resources.items():
            for ref in resource.ResourceGroupReferences:
                self.instance_resource_groups[ref].append(resource)

    def _parse_events(self, XMLEvents: ET.Element):
        XMLEventGroups = XMLEvents.find("EventGroups")
        self.EventGroups = {
            XMLEventGroup.attrib["Id"]: XHSTTSInstance.EventGroup(
                Name=XMLEventGroup.find("Name").text,
                Type=XMLEventGroup.tag,
            )
            for XMLEventGroup in (
                XMLEventGroups.findall("Course") + XMLEventGroups.findall("EventGroup")
            )
        }

        self.Events = {
            XMLEvent.attrib["Id"]: XHSTTSInstance.Event(
                Reference=XMLEvent.attrib["Id"],
                Name=XMLEvent.find("Name").text,
                Duration=int(XMLEvent.find("Duration").text),
                Workload=int(XMLEvent.find("Workload").text)
                if XMLEvent.find("Workload") is not None
                else None,
                PreAssignedTimeReference=XMLEvent.find("Time").attrib["Reference"]
                if XMLEvent.find("Time") is not None
                else None,
                Resources=[
                    XHSTTSInstance.EventResource(
                        Reference=XMLResource.attrib.get(
                            "Reference"
                        ),  # can be None in which case the solver is expected to assign a resource to the event and in this case the Role and ResourceType are compulsory
                        Role=XMLResource.find("Role").text
                        if XMLResource.find("Role") is not None
                        else None,
                        ResourceTypeReference=XMLResource.find("ResourceType").attrib[
                            "Reference"
                        ]
                        if XMLResource.find("ResourceType") is not None
                        else None,
                        Workload=int(XMLResource.find("Workload").text)
                        if XMLResource.find("Workload") is not None
                        else None,
                    )
                    for XMLResource in XMLEvent.find("Resources").findall("Resource")
                ],
                ResourceGroupReferences=[
                    XMLResourceGroup.attrib["Reference"]
                    for XMLResourceGroup in XMLEvent.find("ResourceGroups").findall(
                        "ResourceGroup"
                    )
                ]
                if XMLEvent.find("ResourceGroups") is not None
                else [],  # every resource in the resource group is to be preassigned to the event, however spec may change as was intended to be used for student sectioning
                # TODO: do any datasets use this?
                CourseReference=XMLEvent.find("Course").attrib["Reference"]
                if XMLEvent.find("Course") is not None
                else None,
                EventGroupReferences=[
                    XMLEventGroup.attrib["Reference"]
                    for XMLEventGroup in XMLEvent.find("EventGroups").findall(
                        "EventGroup"
                    )
                ]
                if XMLEvent.find("EventGroups") is not None
                else [],
            )
            for XMLEvent in XMLEvents.findall("Event")
        }

        # populate instance_event_groups
        for _, event in self.Events.items():
            for ref in event.EventGroupReferences:
                self.instance_event_groups[ref].append(event)
            if event.CourseReference:
                self.instance_event_groups[event.CourseReference].append(event)

    def _parse_constraints(self, XMLConstraints: ET.Element):
        for XMLConstraint in XMLConstraints.findall("*"):
            class_name = XMLConstraint.tag
            try:
                constraint_class = globals()[class_name]
            except KeyError:
                raise Exception(f"Unrecognized constraint: {class_name}")
            assert callable(constraint_class) and isclass(
                constraint_class
            ), f"Exception: {class_name} is not a valid class"
            constraint_instance = constraint_class(
                XMLConstraint,
                self.instance_resource_groups,
                self.instance_event_groups,
                self.instance_time_groups,
                self.Events,
            )
            self.Constraints.append(constraint_instance)

    def _parse_solutions(self, XMLSolutions: list[ET.Element]):
        for XMLSolution in XMLSolutions:
            solution_events = XMLSolution.find("Events").findall("Event")
            self.Solutions.append(
                [
                    XHSTTSInstance.SolutionEvent(
                        InstanceEventReference=XMLSolutionEvent.attrib["Reference"],
                        Duration=int(XMLSolutionEvent.find("Duration").text)
                        if XMLSolutionEvent.find("Duration") is not None
                        else self.Events[XMLSolutionEvent.attrib["Reference"]].Duration,
                        TimeReference=XMLSolutionEvent.find("Time").attrib["Reference"]
                        if XMLSolutionEvent.find("Time") is not None
                        else None,
                        Resources=[
                            XHSTTSInstance.SolutionEventResource(
                                XMLSolutionResource.attrib["Reference"],
                                XMLSolutionResource.find("Role").text,
                            )
                            for XMLSolutionResource in XMLSolutionEvent.find(
                                "Resources"
                            ).findall("Resource")
                        ]
                        if XMLSolutionEvent.find("Resources") is not None
                        else [],
                    )
                    for XMLSolutionEvent in solution_events
                ]
            )

    def get_cost(self, solution, constraint: Constraint):
        return constraint.evaluate(solution)

    def evaluate_solution(self, solution: list[SolutionEvent]):
        cost = Cost(Infeasibility_Value=0, Objective_Value=0)

        for constraint in self.Constraints:
            value = self.get_cost(solution, constraint)
            # print("value = ", value, constraint.__class__.__name__)
            if constraint.is_required():
                cost.Infeasibility_Value += value
            else:
                cost.Objective_Value += value

        return cost

    # TODO : make list -> list
    @staticmethod
    def create_solution_event(event: Event) -> SolutionEvent:
        return XHSTTSInstance.SolutionEvent(
            InstanceEventReference=event.Reference,
            Duration=event.Duration,
            TimeReference=event.PreAssignedTimeReference,
            Resources=[
                XHSTTSInstance.SolutionEventResource(
                    Reference=resource.Reference, Role=resource.Role
                )
                for resource in event.Resources
            ],
        )


class XHSTTS:
    def __init__(self, filename: str):
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        self.instances = self.root.findall(".//Instance")

    def get_first_instance(self):
        return XHSTTSInstance(
            self.instances[0],
            self.root.findall(
                f".//SolutionGroups/SolutionGroup//Solution[@Reference='{self.instances[0].attrib['Id']}']"
            ),
        )

    def get_instance(self, index: int):
        assert index < len(
            self.instances
        ), "index should be less than the number of instances."
        assert index >= 0, "index should be greater than or equal to 0."
        return XHSTTSInstance(
            self.instances[index],
            self.root.findall(
                f".//SolutionGroups/SolutionGroup//Solution[@Reference='{self.instances[0].attrib['Id']}']"
            ),
        )

    def num_instances(self):
        return len(self.instances)


if __name__ == "__main__":
    first_instance = XHSTTS(
        "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/ArtificialAbramson15.xml"
    ).get_first_instance()

    import re

    def get_event_number(event):
        event_str = str(event)
        pattern = r"Event_C(\d+)_\d+"
        match = re.search(pattern, event_str)

        if match:
            number = match.group(1)
            return int(number)
        else:
            raise Exception(f"Number not found in the event. Event = {event_str}")

    print(list(map(get_event_number, first_instance.get_solutions()[1])))

    one_to_450 = {x for x in range(1, 450 + 1)}

    print(
        "solution 0",
        len(list(map(get_event_number, first_instance.get_solutions()[0]))),
        one_to_450 - set(map(get_event_number, first_instance.get_solutions()[0])),
    )

    print(
        "solution 1",
        len(list(map(get_event_number, first_instance.get_solutions()[1]))),
        one_to_450 - set(map(get_event_number, first_instance.get_solutions()[1])),
    )

    path = (
        "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/BrazilInstance3.xml"
    )

    # path = (
    #     "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/AustraliaBGHS98.xml"
    # )

    # path = "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/ArtificialSudoku4x4.xml"

    first_instance = XHSTTS(path).get_first_instance()
    print(first_instance.get_times())
    print(first_instance.get_resources())
    print(first_instance.ResourceTypes)
    print(first_instance.ResourceGroups)
    print(first_instance.get_constraints())
