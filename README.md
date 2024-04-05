# RouteExplainer: An Explanation Framework for Vehicle Routing Problem (Work in progress)
<p align="center">
  <a href="https://ntt-dkiku.github.io/xai-vrp/" target="_blank"><img src="https://img.shields.io/badge/project-page-blue"></a>
  <a href="https://arxiv.org/abs/2403.03585" target="_blank"><img src="https://img.shields.io/badge/arXiv-abs-red"></a>
  <a href="https://huggingface.co/spaces/oookiku/route-explainer" target="_blank"><img src="https://img.shields.io/badge/🤗-demo-yellow"></a>
  <a href="https://pakdd2024.org/" target="_blank"><img src="https://img.shields.io/badge/PAKDD-2024-green"></a>
</p>
This repo is the official implementation of <a href="https://arxiv.org/abs/2403.03585" target="_blank">RouteExplainer: An Explanation Framework for Vehicle Routing Problem</a> (PAKDD 2024).

## 📦 Setup
We recommend using Docker to setup development environments. Please use the [Dockerfile](./Dockerfile) in this repository. 
In the following, all commands are supposed to be run inside of the Docker container.
```
docker build -t route_explainer/route_explainer:1.0 .
```
You can run code interactively in the container after launching the container by the following command (<> indicates a placeholder, which you should replace according to your settings).
Please set the ```shm_size``` as large as possible because the continuous reuse of conventional solvers (e.g., Concorde and LKH) consumes a lot of shared memory.
The continuous reuse is required when generating datasets and evaluating edge classifiers.  
```
docker run -it --rm -v </path/to/clone/repo>:/workspace/app --name evrp-eps -p <host_port>:<container_port> --shm-size <large_size (e.g., 30g)> --gpus all route_explainer/route_explainer:1.0 bash
```
If you use LKH and Concorde, you need to install them by the following command. LKH and Concorde is required for reproducing experiments, but not for demo.
```
python install_solvers.py
```

## 🧪 Reproducibility
<!-- Refer to [reproduce_experiments.ipynb](./reproduct_experiments.ipynb). -->
Coming Soon!

## 🔧 Training and evaluating edge classifiers
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

## 💬 Explanation generation (demo)
Go to <a href="https://localhost:<container_port>" target="_blank">localhost:<container_port></a> after launching the streamlit app by the following command.  
This is a standalone demo, so you may skip the above experiments and try this first.
```
streamlit run app.py --server.port <container_port>
```
We also publish this demo on Hugging Face Spaces, so you can easily try it <a href="https://huggingface.co/spaces/oookiku/route-explainer" target="_blank">there</a>.


## 🐞 Bug reports and questions
If you encounter a bug or have any questions, please post issues in this repo.

## 📄 Licence
Our code is licenced by NTT. Basically, the use of our code is limitted to research purposes. See [LICENSE](./LICENSE) for more details.

## 🤝 Citation
```
@article{dkiku2024routeexplainer,
  author = {Daisuke Kikuta and Hiroki Ikeuchi and Kengo Tajiri and Yuusuke Nakano},
  title = {RouteExplainer: An Explanation Framework for Vehicle Routing Problem},
  year = 2024,
  journal = {arXiv preprint arXiv:2403.03585}
  url = {https://arxiv.org/abs/2403.03585}
}
```
