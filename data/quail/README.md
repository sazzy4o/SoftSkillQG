# QuAIL: Question Answering for Artificial Intelligence

This repository contains the main and challenge data for QuAIL reading comprehension dataset. QuAIL contains 15K multi-choice questions in texts 300-350 tokens long 4 domains (news, user stories, fiction, blogs). QuAIL is **balanced and annotated for question types**:
 
 > - **text-based questions** (the information should be found in the text):
>   * reasoning about factual information in the text (e.g. *What did John do?*);
>   * temporal order of events (e.g. *When did X happen - before, after, or during Y?*);
>   * character identity (e.g. *Who ate the cake?*);
> - **world knowledge questions** (rely on some kind of inference about characters and events, based on information in the text and world knowledge):
>   * causality (e.g. *Why did John eat the cake?*);
>   * properties and qualities of characters (e.g. *What kind of food does John probably like?*);
>   * belief states (e.g. *What does John think about Mary?*); 
>   * subsequent state after the narrated events (e.g. *What does John do next?*);
>   * duration of the narrated events (e.g. *How long did it probably take John to eat the cake?*);
> - **unanswerable questions** are questions for which the information is not found in the text, and all answer options are equally likely (e.g. *What is John's brother's name?*)

See the blog post describing the project and updates post-paper-publication: https://text-machine-lab.github.io/blog/2020/quail

The data is available in [NLP Datasets library from Hugging Face](https://huggingface.co/nlp/viewer/?dataset=quail), as well as in this repository. We provide two formats: xml and jsonlines. The json format is what is used in our [leaderboard](http://text-machine.cs.uml.edu/lab2/projects/quail/); you can get the files [here](quail_v1.3/json) or in the [codalab data sheet](https://worksheets.codalab.org/worksheets/0xa8cd6ea812c04be7b728f44b2e0a56fc). The xml format is more human-readable, and there are two versions of each file: 

* [randomized](quail_v1.3/xml/randomized): the order of questions and answers is randomized. This is the version recommended for use in training/testing models, and the order is the same as the version provided in the Hugging Face Datasets collection
 * [ordered](quail_v1.3/xml/ordered): the questions are ordered by type, and the correct answers are listed first. This is the more human-readable version for data exploration. 
 
 The order of the texts in the same in both sets. The jsonl files are only provided in the randomized version. See the results of experiments with different data splits [here](https://text-machine-lab.github.io/blog/2020/quail/#update-vaiance-in-fine-tuning). 

For each format we provide three data splits:

* train (560 texts in 4 genres, 10346 questions)
* dev (120 texts in 4 genres, 2164 questions)
* challenge set: (30 fiction texts, 556 questions, written to be difficult for NLP systems that learned to rely on naive lexical matching. Here you will find [the data](quail_v1.3/xml/randomized/quail_1.3_challenge_randomized.xml) and our [notes](quail_v1.3/QuAIl_challenge_set.md) with different paraphrasing strategies. 

Published paper: https://aaai.org/ojs/index.php/AAAI/article/view/6398

Leaderboard: http://text-machine.cs.uml.edu/lab2/projects/quail/

Leaderboard submission instructions: https://worksheets.codalab.org/worksheets/0xcb1727289e2b4688b6534c8253b3bc8e

***********************

Update 30.10.2020: QuAIL v.1.3: the data is now available in [NLP Datasets](https://huggingface.co/nlp/viewer/?dataset=quail), and we switched to jsonlines to match the format to the online HuggingFace data viewer. Question order within texts and answer order within questions changed (beware of Python's random compatibility across systems and python versions [[REF]](https://stackoverflow.com/questions/8786084/reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions)).

