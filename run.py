"""
Author: Timothy Ruiz Docena
Date: September 2024

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/4.0/


Description:
This code is part of the Spanish Sign Language gesture recognition project.
It runs the scripts to recognise Spanish Sign Language gestures using CNN and SVM 
pre-trained models.
"""

import os
import sys


def list_files(directory):
    # Lists all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files


def select_file(files):
    # Displays a numbered list of the files for the user to select from
    print("Select a file to run:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    # Requests the user to select a file
    while True:
        try:
            selection = int(input("Enter the file number: ")) - 1
            if selection < 0 or selection >= len(files):
                print("Invalid selection, try again.")
            else:
                return files[selection]
        except ValueError:
            print("Please enter a valid number.")


def run_script(script_name,file_name):
    """ Runs a Python script passing the filename as argument """
    os.system(f"{sys.executable} {script_name} {file_name}")


def select_script(file_selected):
    """ Selects the script to run according to the type of model """
    scripts = {
        "keras": os.path.abspath(os.path.join(__file__,"..","CNN","predict.py")),
        "pkl": os.path.abspath(os.path.join(__file__,"..","SVM","predict.py"))
    }
    termination = file_selected.split(".")[-1]
    return scripts[termination]


if __name__ == "__main__":
    # Relative directory where to look for the files
    models_directory = os.path.abspath(os.path.join(__file__,"..","models"))

    # List files in the directory
    files = list_files(models_directory)

    if not files:
        raise Exception("No files were found in the directory.")

    file = select_file(files)
    # Select a file from the list
    file_selected = os.path.join(models_directory,file)
    script = select_script(file_selected)

    # Execute the selected script
    run_script(script,file_selected)
