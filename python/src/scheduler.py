import xml.etree.ElementTree as ET

# Define the path to your XHSTTS dataset XML file
xml_file_path = "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/ArtificialAbramson15.xml"

# Parse the XML file
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Iterate through 'Instance' elements
for instance in root.findall(".//Instance"):
    instance_id = instance.attrib["Id"]
    instance_metadata = instance.find("MetaData")
    instance_name = instance_metadata.find("Name").text

    # Access other metadata or elements as needed
    times = instance.find("Times")
    resources = instance.find("Resources")
    events = instance.find("Events")
    constraints = instance.find("Constraints")

    # Print or process data as necessary
    print(f"Instance ID: {instance_id}")
    print(f"Instance Name: {instance_name}")
    print(times, resources, events, constraints)

    # Example: To access specific data from 'Times'
    for time in times.findall("Time"):
        # Access time-related data as needed
        print(time.findall(".//Day || .//Week || .//TimeGroup"))

    # Example: To access specific data from 'Resources'
    for resource in resources.findall("Resource"):
        # Access resource-related data as needed
        pass

    # Example: To access specific data from 'Events'
    for event in events.findall("Event"):
        # Access event-related data as needed
        pass

    # Example: To access specific data from 'Constraints'
    for constraint in constraints.findall("Constraint"):
        # Access constraint-related data as needed
        pass

    print("\n")

# Iterate through 'SolutionGroup' elements
for solution_group in root.findall(".//SolutionGroup"):
    solution_group_metadata = solution_group.find("Metadata")
    # Access metadata as needed

    # Iterate through 'Solution' elements within 'SolutionGroup'
    for solution in solution_group.findall("Solution"):
        # Access solution-related data as needed

        # Print or process data as necessary
        pass
