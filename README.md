# Data-Acquisition-Tool-for-Solar-Radio-Observations

This project explores the detection and data acquisition of electromagnetic waves. A component of this work involves the acquisition of real-time data EM data using analog-to-digital converters (ADCs), which apply the principles of sampling theory and signal processing. 

**Project Overview**

This repository presents a Python-based system for acquiring and visualizing electromagnetic (EM) signals from solar radio activity in real time. It integrates:

-Antenna theory and design for radio frequency (RF) acquisition

-Signal processing techniques for real-time data interpretation

-A Graphical User Interface (GUI) for monitoring and control

-Data storage in FITS and PNG formats for scientific analysis

**Objectives**

-Detect solar radio signals using an appropriate antenna setup

-Acquire data using ADCs and sampling theory

-Process and visualize EM signals as a real-time spectrogram

-Store scientific data in flexible formats for future analysis

-Make the system modular and user-friendly for researchers and learners

**Features**

-Real-time scrolling spectrogram viewer

-Adjustable sampling frequency and FFT size

-Live average peak frequency display

-Save data periodically in FITS and PNG formats

-Color map selection and intensity range sliders

-Supports multithreading for a responsive user experience

-TCP-based data acquisition or simulated signal input

**Technologies & Libraries**

Python 3.8+

Tkinter – GUI design

Matplotlib – Spectrogram plotting

NumPy – Numerical processing

Astropy – FITS file handling

Threading – Asynchronous processing

Socket – TCP communication

The file plot_inno2.py uses randomly generated data while the file plot_tcp takes data in real-time through an integrated TCP port. 
