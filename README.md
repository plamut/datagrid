Datagrid
========

Install (requires Python 3.3+)
------------------------------

    $ git clone git@github.com:plamut/datagrid.git
    $ cd datagrid
    $ virtualenv -p python3.3 ./
    $ source bin/activate

Run
---

    (datagrid)$ python src/main.py

*Note:* Simulation parameters can be changed in *main.py*.

Run tests
---------
    (datagrid)$ python -m unittest discover -v -s src/
