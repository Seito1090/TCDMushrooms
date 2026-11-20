# TCDMushrooms
ST3005 INFORMATION SYSTEMS Group Project Repo

## General information

Selected topic : Will this mushroom kill you ? 

Expected output : Yes/No 

Unwanted output : Maybe 

## Structure of the repo 

For ease of maintenance, it would be nice to separate different aspects of the project. 

With this in mind, the main directories would be : 

- data/ : data sets used, format to be determined.
- graphs/ : generated graphs by the code
- presentation/ : slides and everything else needed for the presentation
- report/ : same as presentation/ but for the report
- src/ : code used in the project 

## Data sets sources  
-  https://www.openml.org/search?type=data&sort=runs&id=24&status=active

## How to run

We follow the same principle as the practicals in class, using a virtual environment. 

The required packages are : 
- numpy
- matplotlib
- seaborn
- scipy
- pandas 

You can either set up the venv yourself, or use the script for your system
- linux : ``` bash setup.sh ```
- windows : ``` .\setup.bat ``` 

Note : If you do not have the data and / or the graphs directory, create it 
