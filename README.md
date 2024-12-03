# SoftSkillQG

Tool for automatically generating skill targeted reading comprehension questions. 

## Setup/install
The code was made to run on the Compute Canada (Gentoo Linux) environment. Some modifications may be required to run in other environments

On Compute Canada run:

```
source ./activate.sh 
```

## Running
You can run each file like:
```
python3 A_train_question_generator_control.py single_model_with_soft_prompt_t5_lm 5e-5 quail large 1
```
Some programs can also be run as vscode notebooks.

The interface for each program is not fully documented. If you are having problems, please create a GitHub issue.

## License
Most of the code in this repo is licensed until the Apache 2.0 license. Files in some folders have a difference license, please refer to the licenses in these folders for more details.

## Citations
Cite as:
```

@article{von_der_Ohe_Fyshe_SoftSkillQG,
	author = {von der Ohe, Spencer McIntosh and Fyshe, Alona},
	journal = {Proceedings of the Canadian Conference on Artificial Intelligence},
	year = {2024},
	month = may,
    url = {https://caiac.pubpub.org/pub/n3kbetb8},
	publisher = {Canadian Artificial Intelligence Association (CAIAC)},
	title = {SoftSkillQG: Robust Skill-Targeted Reading Comprehension Question Generation Using Soft-prompts},
}
```
