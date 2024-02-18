# Risk-aware-contingency-planning-with-multi-modal-predictions


## General Info
This repository includes the source code associated with the paper "RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions". The paper presents an approach that leverages Bayesian beliefs over the distribution of potential policies of
other road users to construct a novel risk-aware probabilistic motion planning framework. In particular, we propose a novel contingency planner that outputs long-term contingent plans conditioned on multiple possible intents for other actors in the traffic scene. The Bayesian belief is incorporated into the optimization cost function to influence the behavior of the short-term plan based on the likelihood of other agents’ policies. Furthermore, a probabilistic risk metric is employed to fine-tune the balance between efficiency and robustness. The algorithm is designed to work seamlessly with the [CommonRoad](https://commonroad.in.tum.de/) simulation environment and adopts the Frenet planner base code from [this repository](https://github.com/TUMFTM/EthicalTrajectoryPlanning/tree/master).

## Requirements
The software has been tested on the following setup:
* Ubuntu 20.04.
* Python 3.8.
* The required python dependencies are specified in the requirements.txt.

Please note that it is recommended to create an isolated virtual environment where the required Python packages can be installed.

## Installation
To run this code, please follow this set of instructions:
* Clone this repository
```
$ git clone https://github.com/KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions.git
```
* Navigate to the root folder of the repository and install the dependencies
```
pip install -r requirements.txt
```
* To run the contingency planner, execute the following command from the root directory
```
python planner/Frenet/frenet_planner.py
```

