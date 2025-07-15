import json

def group_similar_videos(json_file_path):
    """
    Groups videos based on similarity percentages from a JSON file.

    Args:
        json_file_path (str): Path to the input JSON file.

    Returns:
        str: JSON string representing the grouped videos.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": "Input file not found."}, indent=2)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format in input file."}, indent=2)

    output_groups = []
    for tag_name, comparisons in data["comparisons"].items():
        similar_videos_in_tag = set()
        for comp in comparisons:
            if comp["videoNamesSimilarityPercentage"] >= 90:
                similar_videos_in_tag.add(comp["inspectedVideo"])
                similar_videos_in_tag.add(comp["referenceVideo"])
        if similar_videos_in_tag:
            output_groups.append({"videos": list(similar_videos_in_tag), "tags": [tag_name]})

    return json.dumps(output_groups, indent=2)

if __name__ == "__main__":
    input_file = "video_names.json"  # Name of the input JSON file
    output_json = group_similar_videos(input_file)
    print(output_json)