from collections import namedtuple
import xml.etree.ElementTree as ET

# Hack, TODO sort out modules properly! (virtual env? Add to path?)
try:
    from .utils import Cost
except:
    from utils import Cost


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
            "PreAssignedTimeReferences",
            "Resources",  # of type EventResource
            "ResourceGroupReferences",
            "CourseReference",
            "EventGroupReferences",
        ],
    )
    EventResource = namedtuple(
        "EventResource", ["Reference", "Role", "ResourceTypeReference", "Workload"]
    )

    def __init__(self, XMLInstance, XMLInstanceSolutions):
        self.TimeGroups = {}
        self.Times = {}
        self.ResourceTypes = {}
        self.ResourceGroups = {}
        self.Resources = {}
        self.Events = {}
        self._parse_times(XMLInstance.find("Times"))
        self._parse_resources(XMLInstance.find("Resources"))
        self._parse_events(XMLInstance.find("Events"))
        self._parse_constraints(XMLInstance.find("Constraints"))

    def get_events(self):
        return self.Events

    def get_times(self):
        return self.Times

    def get_resources(self):
        return self.Resources

    def get_constraints(self):
        return self.constraints

    def get_solutions(self):
        return self.solutions

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
                PreAssignedTimeReferences=XMLEvent.find("Time").attrib["Reference"]
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
                        Workload=int(XMLResource.find("Workload"))
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
                ],
            )
            for XMLEvent in XMLEvents.findall("Event")
        }

    def _parse_constraints(self, XMLConstraints: ET.Element):
        pass

    @staticmethod
    def get_cost(solution, constraint):
        return

    @staticmethod
    def evaluate_solution(solution, constraints):
        cost = Cost(Infeasibility_Value=0, Objective_Value=0)

        for constraint in constraints:
            value = XHSTTSInstance.get_cost(solution, constraint)
            if constraint.is_required():
                cost.Infeasibility_Value += value
            else:
                cost.Objective_Value += value

        return Cost


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
        "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/BrazilInstance3.xml"
    ).get_first_instance()
    # print(first_instance.get_times())
    # print(first_instance.get_resources())
    # print(first_instance.ResourceTypes)
    # print(first_instance.ResourceGroups)

    from pprint import pp

    pp(list(first_instance.get_events().items())[:5])
