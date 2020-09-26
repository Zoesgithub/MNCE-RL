# Reinforced Molecular Optimization with Neighborhood-Controlled Grammar

![Illustration of Our Framework.](./Image/framework.png)

## Requirements

Anaconda is recommended to run the project.
~~~
conda create -n MNCERL python=3.6 
source activate MNCERL
~~~

Install rdkit:
~~~
conda install -c conda-forge rdkit
~~~

Install related packages:
~~~
pip install -r requirement.txt
cd MyLib
python setup.py install
~~~
Prepare data:
~~~
cd Data
ls *.tar.gz|while read line
do
tar -xzvf ${line}
done
~~~

## Training and evaluations

You can run training and evaluations by:
~~~
python main.py -c PATH_TO_CONFIG
~~~
Please refer to *config_example.py* for the format of the config file. In the tasks directory, we have provided the pretrained model, and the *config.py* and results for all the tasks presented in our paper.
