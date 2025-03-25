"""
Author: Timothy Ruiz Docena
Date: September 2024

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/4.0/


Description:
This code is part of the Spanish Sign Language gesture recognition project.
It runs the script to install dependencies and runs the gesture recognition script.
"""

import os
import subprocess
import sys

def create_virtualenv(venv_name):
    """ Create a virtual environment """
    if not os.path.exists(venv_name):
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])

def install_requirements(venv_name):
    """ Install packages from requirements.txt in the virtual environment """
    pip_path = os.path.join(venv_name, "Scripts", "pip.exe")
    os.system(os.path.join(venv_name, "Scripts", "activate"))
    os.system(" ".join([pip_path, "install", "-r", "requirements.txt"]))

def run_script(venv_name, script_name):
    """ Run the script inside the virtual environment """
    python_path = os.path.join(venv_name, "Scripts", "python.exe")
    os.system(" ".join([python_path, script_name]))

if __name__ == "__main__":
    venv_name = "venv"
    script_name = "run.py"

    print("Creating virtual environment")
    create_virtualenv(venv_name)
    print("Installing requirements")
    install_requirements(venv_name)
    print("Running...")
    run_script(venv_name, script_name)
