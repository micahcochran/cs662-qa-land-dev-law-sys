
# Zoning Ordinance Question Answer

Group Members: Micah Cochran and Seth Lewis

## Prerequisites
* ~~Anaconda w/ Python~~
* Python

### Pre-install instructions Debian Linux
From root account:
```bash
$ sudo apt-get install git python3-pip python3-venv python3-wheel
```

## Installation
1. Either clone the GitHub Repository:  
```
git clone https://github.com/micahcochran/cs662-qa-land-dev-law-sys.git
```
OR download the ZIP file from GitHub.

2. ~~Install the conda environment. Installation Details in [env/](env/).  Note this process may take a while.~~

Read the README in [env/](env/) for next steps in installation.


### Command Line Interface
`kgqas/cli.py` is a command line interface for the Knowledge Graph Question Answering system version. 

## Folders

* [env/](env/) - [environment](./env/ENV.md), files to create the environment.
* [kgqas/](kgqas/) - Zoning Information Knowledge Graph Question Answering System. Go here to run jupyter notebook ([kgqas/KGQAS.ipynb](kgqas/KGQAS.ipynb)).
* [triplesdb/](triplesdb/)  
    - triples knowledge graph "database" is stored here in [triplesdb/combined.ttl](triplesdb/combined.ttl)
    - [triplesdb/generate_template.py](triplesdb/generate_template.py) generates questions for training the models.
    - the `.rq` files are sample SPARQL queries used during development.  Read [triplesdb/README.md](triplesdb/README.md) for more information about running such queries.

* [programs/](programs/) - [see PROGRAMS.md](./programs/PROGRAMS.md), integration tests and jupyter notebooks. Go here for example code - **CURRENTLY NOT WORKING WITH VENV environment.**
* [proposal/](proposal/) - This was a proposal of the work before the project.  There are some thoughts about how to approach the domain and how we initially thought about approaching the project.

## Attribution

Attribution of work:
* nlp/ - corpus loading code was Seth's work, Zoning Ordinance text was Micah's contribution 
* kgqas/ - Micah
* programs/ - Seth
* proposal/ - Micah and Seth 
* poster/ - Micah's design and Seth contributed his work portions
* tripeldb/ - Micah