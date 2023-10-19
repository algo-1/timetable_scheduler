import xml.etree.ElementTree as ET


class XHTTS:
    def __init__(self, filename: str):
        self.Tree = ET.parse(filename)
        self.TIME_GROUP_NAMES = ["TimeGroup", "Day", "Week"]
        self.EVENT_GROUP_NAMES = ["EventGroup", "Course"]
        self.CONSTRAINT_NAMES = [
            "AssignResourceConstraint",
            "AssignTimeConstraint",
            "PreferResourcesConstraint",
            "AvoidClashesConstraint",
        ]
