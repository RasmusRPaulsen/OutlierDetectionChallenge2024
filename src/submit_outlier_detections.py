import argparse
import os
import pathlib
from datetime import datetime
import re
import requests
from dtu_spine_config import DTUConfig
import socket
import json
from pathlib import Path

def sanitize_string(string_in):
    """
    Will replace bad characters with _
    """
    string_in = os.path.basename(string_in)
    string_in = os.path.splitext(string_in)[0]
    string_in = string_in.strip().replace(':', '_')
    string_in = string_in.strip().replace('-', '_')
    string_in = string_in.strip().replace('.', '_')
    re.sub(r'[^\w\-_\. ]', '_', string_in)
    return string_in


def submit_outlier_detections(settings):
    result_dir = settings["result_dir"]
    test_results_json = os.path.join(result_dir, "test_results.json")
    data_set = settings["data_set"]
    team_name = settings["team_name"]
    method_name = settings["method_description"]
    data_types = settings["data_types"]
    server_name = settings["challenge_server"]
    submission_dir = os.path.join(result_dir, "submissions")
    pathlib.Path(submission_dir).mkdir(parents=True, exist_ok=True)

    # team_name = sanitize_string(team_name)
    # method_name = sanitize_string(method_name)
    # timestamp = datetime.now().strftime(r'%d%m%y_%H%M%S')
    timestamp = datetime.now().strftime(r'%d%m%y_%H%M%S')
    pc_name = socket.gethostname()
    pc_name= pc_name.lower()

    submission_file = os.path.join(submission_dir, f"{pc_name}-{timestamp}.json")
    print(f"Creating {submission_file} from {test_results_json}")
    if not os.path.exists(test_results_json):
        print(f"Error: {test_results_json} does not exist")
        return
    if team_name == "ChangeYourTeamName":
        print("Error: Please change the team name in the config file - it is currently set to 'ChangeYourTeamName'")
        return

    try:
        with open(test_results_json, 'r') as openfile:
            test_results = json.load(openfile)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {test_results_json}")
        return

    submission_dict = {}
    submission_dict["team_name"] = team_name
    submission_dict["method_name"] = method_name
    submission_dict["timestamp"] = datetime.now().strftime("%c")
    submission_dict["data_set"] = data_set
    submission_dict["data_types"] = data_types
    submission_dict["results"] = test_results

    try:
        with Path(submission_file).open('wt') as handle:
            json.dump(submission_dict, handle, indent=4, sort_keys=False)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {submission_file}")
        return

    print(f"Uploading to server: {server_name}")
    file_name = f"{submission_file}"
    test_file = open(file_name, "rb")
    test_url = server_name
    try:
        test_response = requests.post(test_url, files={"form_field_name": test_file})
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return

    if test_response.ok:
        print("Upload completed successfully!")
        print(test_response.text)
    else:
        print("Something went wrong!")
        print(test_response.text)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='submit-outlier-detection')
    config = DTUConfig(args)
    if config.settings is not None:
        submit_outlier_detections(config.settings)
