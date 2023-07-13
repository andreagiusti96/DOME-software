# DOME-software

This software allows the use of the DOME (Dynamic Optical Micro-Environment).
For more info about the DOME visit http://theopendome.org.

To start your first experiment with the DOME run DOME_experiment.py on the Raspberry Pi 4 connected to the camera.

Features:
  - Acquire images and video from the DOME camera (see DOME_experiment.py and DOME_microscope.py)
  - Project configurable and time varying light inputs and patterns (see DOME_experiment.py)
  - Automatic image analysis to detect and track the agents (see DOME_tracker.py)
  - Calibration to map the camera and the projector frames (see DOME_calibration_camera(_v2).py)

Main scripts:
  - DOME_experiment: acquire video and images while controlling the projector. Use the console to input single commands or define an experiment as a sequence of actions. This is the main script to use the DOME.
  - DOME_calibration: automatic calibration of the camera-to-projector transformation.
  - DOME_calibration_v2: manual calibration of the camera-to-projector transformation.
  - DOME_tracker: reads the images acquired during an experiment to detect and track the agents.
  - DOME_experiment_analysis: analyse the data generated by DOME_tracker.

Utility scripts:
  - DOME_graphics: utility for plots and images manipulation.
  - DOME_imaging_utilities: utility to use the camera.
  - DOME_communication: utility to manage the wifi communication between the two Raspberries Pi.
  - DOME_experiment_manager: utility to manage the storage of data generated during an experiment.
  - DOME_transformation: utility to manage the camera-to-projector transformation and the projection of shapes.
  - DOME_projection_interface: to run on the Raspberry Pi connected to the projector. Recives the commands from the Raspberry Pi connected to the camera and controls the projector.