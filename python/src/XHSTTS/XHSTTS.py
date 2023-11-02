from __future__ import annotations
from collections import namedtuple
from inspect import isclass
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod


# Hack, TODO sort out modules properly! (virtual env? Add to path? setup.py?)
try:
    from .utils import Cost, Cost_Function_Type, cost_function_to_enum, cost_function
except:
    from utils import Cost, Cost_Function_Type, cost_function_to_enum, cost_function


class Constraint(ABC):
    def __init__(self, XMLConstraint: ET.Element):
        self.name: str = XMLConstraint.find("Name").text
        self.required: bool = XMLConstraint.find("Required").text == "true"
        self.weight = int(XMLConstraint.find("Weight").text)
        self.cost_function: Cost_Function_Type = cost_function_to_enum(
            XMLConstraint.find("CostFunction")
        )
        # TODO - handle appliesTo- create self.events & self.resources

    @abstractmethod
    def evaluate(self, solution: list[XHSTTSInstance.SolutionEvent]):
        pass

    def is_required(self):
        return self.required

    def get_name(self):
        return self.name

    def get_cost_function(self):
        return self.get_cost_function

    def get_weight(self):
        return self.weight


class AssignTimeConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        deviation = 0
        for event in self.events:
            if not event.PreAssignedTimeReference:
                for solution_event in solution:
                    deviation += (
                        (
                            solution_event.Duration
                            if solution_event.Duration
                            else event.Duration
                        )
                        if not solution_event.TimeReference
                        else 0
                    )
        return cost_function(deviation, self.cost_function)


class AssignResourceConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)
        self.role = XMLConstraint.find("Role").text

    def isAssigned(
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
                            (
                                solution_event.Duration
                                if solution_event.Duration
                                else event.Duration
                            )
                            if not self.isAssigned(solution_event.Resources)
                            else 0
                        )

        return cost_function(deviation, self.cost_function)


class PreferResourcesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class AvoidClashesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class SplitEventsConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class DistributeSplitEventsConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class PreferTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class SpreadEventsConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class AvoidUnavailableTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class LimitIdleTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class ClusterBusyTimesConstraint(Constraint):
    def __init__(self, XMLConstraint: ET.Element):
        super().__init__(XMLConstraint)

    def evaluate(self, solution):
        pass


class XHSTTSInstance:
    TIME_GROUP_NAMES = ["TimeGroup", "Day", "Week"]
    EVENT_GROUP_NAMES = ["EventGroup", "Course"]
    CONSTRAINT_NAMES = [
        "AssignResourceConstraint",
        "AssignTimeConstraint",
        "PreferResourcesConstraint",
        "AvoidClashesConstraint",
    ]
    Time = namedtuple("Time", ["Name", "TimeGroupReferences"])
    TimeGroup = namedtuple("TimeGroup", ["Name", "Type"])
    EventGroup = namedtuple("EventGroup", ["Name", "Type"])
    ResourceGroup = namedtuple("ResourceGroup", ["Name", "ResourceTypeReference"])
    Resource = namedtuple(
        "Resource", ["Name", "ResourceTypeReference", "ResourceGroupReferences"]
    )
    Event = namedtuple(
        "Event",
        [
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
        self.Constraints: list[Constraint] = []
        self.Solutions = []
        self._parse_times(XMLInstance.find("Times"))
        self._parse_resources(XMLInstance.find("Resources"))
        self._parse_events(XMLInstance.find("Events"))
        self._parse_constraints(XMLInstance.find("Constraints"))
        self._parse_solutions(XMLSolutions)

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
                    Name=Name, TimeGroupReferences=TimeGroup_references
                )

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

    def _parse_constraints(self, XMLConstraints: ET.Element):
        for XMLConstraint in XMLConstraints.findall("*"):
            class_name = XMLConstraint.tag
            try:
                constraint_class = globals()[class_name]
                assert callable(constraint_class) and isclass(
                    constraint_class
                ), f"{class_name} is not a valid class"
                constraint_instance = constraint_class(XMLConstraint)
                self.Constraints.append(constraint_instance)
            except:
                raise Exception(f"Unrecognized constraint: {class_name}")

    def _parse_solutions(self, XMLSolutions: list[ET.Element]):
        for XMLSolution in XMLSolutions:
            solution_events = XMLSolution.find("Events").findall("Event")
            self.Solutions.append(
                [
                    XHSTTSInstance.SolutionEvent(
                        InstanceEventReference=XMLSolutionEvent.attrib["Reference"],
                        Duration=int(XMLSolutionEvent.find("Duration").text)
                        if XMLSolutionEvent.find("Duration") is not None
                        else None,
                        TimeReference=XMLSolutionEvent.find("Time").attrib["Reference"]
                        if XMLSolutionEvent.find("Time")
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
            if constraint.is_required():
                cost.Infeasibility_Value += value
            else:
                cost.Objective_Value += value

        return cost

    # TODO
    def add_solution():
        pass


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


if __name__ == "__main__":
    first_instance = XHSTTS(
        "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/ArtificialAbramson15.xml"
    ).get_first_instance()

    from pprint import pp

    pp(list(first_instance.get_events().items())[:5])

    pp(first_instance.get_solutions()[1])

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
