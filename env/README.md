
# Instructions to Install

These are instructions that should work on many configurations of Linux.  These are the requirements for the `KGQAS/` and the question generation (in `tripledb/`).

The environment will require about 5.2 GB of storage.
Create a python environment (`env-zoqa` or feel free to use another name) this is a venv style:
```bash
python3 -m venv env-zoqa 
```

Activate that environment. 
```bash
source env-zoqa/bin/activate
```

Install the python packages.
```bash
pip3 install -r env/requirements.txt
```

---
## Notes

If you see something similar to this error:
```
LookupError: 
**********************************************************************
  Resource punkt not found.
  Please use the NLTK Downloader to obtain the resource:
```

Run this to install a required part of NLTK
```
python3 -m nltk.downloader punkt
```

---
## Notes about Anaconda Instruction

Problem: The Anaconda instructions would run for an hour and time out not being able to solve the environment.

There are a couple of reasons that were ran into when development:
1. Intel is the default platform for Python libraries.  Trying to support Apple M1 Macs causes you to have to install alternative versions of the libraries.  The libraries have to use BLAS instead of Intel's MKL implementation.  Intel seems to be the default. These use a different configuration to install for the Apple's M# platform.
2. Two different projects were being developed with little overlap.


# INSTRUCTIONS BELOW THIS POINT ARE BROKEN
---

## Conda environment Required to run all code:

To Install After Cloning Repo (in terminal or command line):
* make sure you have anaconda3 installed
* navigate to env folder
* activate your base conda environment
* run: `conda env create`
    * if conda is taking a long time to solve the environment try setting: `conda config --set channel_priority strict` 

### Notes on commented sections of the environment file:

There are several lines commented out with additional information in the environment file, please pay close attention to these comments if you are installing on any OS that is not Intel Linux