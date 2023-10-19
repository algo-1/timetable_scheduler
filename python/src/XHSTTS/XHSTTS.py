from collections import namedtuple
import xml.etree.ElementTree as ET

from utils import Cost, cost


class XHSTTSInstance:
    TIME_GROUP_NAMES = ["TimeGroup", "Day", "Week"]
    EVENT_GROUP_NAMES = ["EventGroup", "Course"]
    CONSTRAINT_NAMES = [
        "AssignResourceConstraint",
        "AssignTimeConstraint",
        "PreferResourcesConstraint",
        "AvoidClashesConstraint",
    ]
    Time = namedtuple("Time", ["Id", "Name", "TimeGroupReferences"])
    TimeGroup = namedtuple("TimeGroup", ["Name", "Type"])

    def __init__(self, XMLInstance, XMLInstanceSolutions):
        self.TimeGroups = {}
        self.Times = []
        self._parse_times(XMLInstance.find("Times"))

    def get_events(self):
        return self.events

    def get_times(self):
        return self.Times

    def get_resources(self):
        return self.resources

    def get_constraints(self):
        return self.constraints

    def get_solutions(self):
        return self.solutions

    def _parse_times(self, XMLTimes):
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
                self.Times.append(
                    XHSTTSInstance.Time(
                        Id=time_Id, Name=Name, TimeGroupReferences=TimeGroup_references
                    )
                )

    def _parse_TimeGroups(self, XMLTimeGroups):
        return {
            element.attrib["Id"]: XHSTTSInstance.TimeGroup(
                Name=element.find("Name").text, Type=element.tag
            )
            for element in XMLTimeGroups
        }

    def _parse_resources(self, XMLResources):
        pass

    def _parse_events(self, XMLEvents):
        pass

    def _parse_constraints(self, XMLConstraints):
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
        "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/SpainSchool.xml"
    ).get_first_instance()
    print(first_instance.get_times())
