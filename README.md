# 🤫 Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory
​
This is the official repository for our paper: "Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory"<br>

Our benchmark ConfAIde evaluates inference-time privacy implications of LLMs, in interactive settings. The benchmark has 4 tiers, and you can find the dataset/scenarios under the `./benchmark` directory.

![Benchmark Description](assets/figure1.png)

Please cite our work if you found the resources in this repository useful:
```
@article{confaide2023,
  author    = {Mireshghallah, Niloofar and Kim, Hyunwoo and Zhou, Xuhui  and Tsvetkov, Yulia and Sap, Maarten and Shokri, Reza and Choi, Yejin},
  title     = {Can LLMs Keep a Secret? Testing Privacy  Implications of Language Models via Contextual Integrity Theory},
  year      = {2023},
}
```
​
## Reproducing the results
​
First, create the conda environment by running:
```bash
conda env create -f environment.yml
```
​
and then activate it:
```bash
conda activate confaide
```

You can run the evaluation by running the following example command
```bash
python eval.py --model gpt-3.5-turbo-0613 --data-tier 1 --n-samples 10
```
This command will run ChatGPT on Tier 1 with 10 sampled results for each data point. You can choose Tiers from `[1, 2a, 2b, 3, 4]`. There are some other options that you can choose at the bottom of `eval.py`.

## Directory structure
```
./agents: You can implement your own agent in this directory.
./benchmark: Data files for all the 4 tiers of the dataset
```

### Adding your own agent

All you need to do is create an agent class with the method `interact()` or `batch_interact()`.

​

