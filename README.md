# RouteExplainer: An Explanation Framework for Vehicle Routing Problem
This repo is the official implementation of "RouteExplainer: An Explanation Framework for Vehicle Routing Problem" (PAKDD 2024). Please check more details at the project page https://ntt-dkiku.github.io/xai-vrp/.

## Setup
We recommend using Docker to setup development environments. Please use the [Dockerfile](./Dockerfile) in this repository. 
```
docker build -t route_explainer/route_explainer:1.0 .
```
If you use LKH and Concorde, you need to install them by typing the following command. LKH and Concorde is required for reproducing experiments, but not for demo.
```
python install_solvers.py
```
In the following, all commands are supposed to be typed inside the Docker container.

## Reproducibility
<!-- Refer to [reproduce_experiments.ipynb](./reproduct_experiments.ipynb). -->
Coming Soon!

## Training and evaluating edge classifiers
### Generating synthetic data with labels
```
python generate_dataset.py --problem tsptw --annotation --parallel
```

### Training
```
python train.py
```

### Evaluation
```
python eval.py
```

## Explanation generation (demo)
Go to http://localhost:8888 after launching the streamlit app with the following command. You may change the port number as you like.
```
streamlit run app.py --server.port 8888
```

## Licence
Our code is licenced by NTT. Basically, the use of our code is limitted to research purposes. See [LICENSE](./LICENSE) for more details.

## Citation
Coming Soon!