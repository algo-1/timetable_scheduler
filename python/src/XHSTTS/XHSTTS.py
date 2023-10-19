import xml.etree.ElementTree as ET


class XHSTTSInstance:
    def __init__(self, XMLInstance, XMLInstanceSolutions):
        self.TIME_GROUP_NAMES = ["TimeGroup", "Day", "Week"]
        self.EVENT_GROUP_NAMES = ["EventGroup", "Course"]
        self.CONSTRAINT_NAMES = [
            "AssignResourceConstraint",
            "AssignTimeConstraint",
            "PreferResourcesConstraint",
            "AvoidClashesConstraint",
        ]
        pass

    def get_events(self):
        pass

    def get_times(self):
        pass

    def get_resources(self):
        pass

    def get_constraints(self):
        pass

    def get_solutions(self):
        pass

    @staticmethod
    def evaluate_solution():
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
    print(first_instance.TIME_GROUP_NAMES)
