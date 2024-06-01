# QuAIL: paraphrasing experiment

1. [Project goals](#project-goals)
4. [Questions that can be answered with the information in the text](#qtext) 
   * [What questions are easy for AI systems?](#qtext_tips)
   * [Questions about temporal order of events](#temporal_order)
   * [Questions about coreference](#coreference)
   * [Factual questions](#factual)
   * [Questions about causality](#causality)
5. [Questions requiring world knowledge](#qworld)
   * [What questions are easy for AI systems?](#qworld-tips)
   * [Questions about subsequent states](#subsequent_state)
   * [Questions about properties of characters and entities](#properties)
   * [Questions about event durations](#event_duration)
   * [Questions about belief states](#information_attribution)         
6. [Unanswerable questions](#unanswerable)

  
# <a id="project-goals"> Project goals </a>

QuAIL project aims to improve the ability of artificial intelligence (AI) systems to answer questions in natural language. In particular, we are focusing on the scenario in which the system reads a short text (about 350 words) and is presented with multiple-choice questions about this text, with the task of selecting the correct answer option out of 4.

There are 3 major types of questions:

1) The questions for which the correct answer can be found in the text directly (with 3 subtypes).
2) The questions for which the correct answer cannot be found in the text directly, but common world knowledge makes one of the answer options much more likely than the others (with 5 subtypes).
3) The questions for which the correct answer cannot be found in the text, and common knowledge does not make one of the answer options more likely than the others.

The questions in QuAIL dataset were written by Amazon Mechanical Turk workers. Each question  has 4 answer options, of which the option "not enough information" is the correct answer for unanswerable questions.  However, many questions are easy for AI systems to answer without any real understanding of the text. This problem persists in many other RC datasets, and inspired a lot of work on adversarial filtering of crowdsourced data; see our [blog post](https://text-machine-lab.github.io/blog/2020/quail/#discussion-other-ways-to-increase-question-diversity) for some references. The general problem with this approach is that data created to be adversarial to one model will not necessarily be equally different for another one, so ideally we would have some model-independent strategies for authoring difficult questions.

In scope of QuAIL project, we conducted an additional experiment with creating an extra challenge: an adversarial set of questions paraphrased from Turker-authored questions and answers. This document is a collection of working notes on strategies we used for writing these more difficult questions. The challenge set contains 556 questions (30 fiction texts, 18 questions each), available in this repository. We found that these questions were indeed significantly more difficult for systems trained on regular QuAIL data. See the results of this experiment [here](https://text-machine-lab.github.io/blog/2020/quail/#paraphrasing-hurts). 
 
 If you find this useful for developing your own RC data, please cite [our paper](https://aaai.org/ojs/index.php/AAAI/article/view/6398) :) 
 
# <a id="qtext"/></a> Questions that can be answered with the information in the text 

QuAIL contains the following types of questions that can be answered with the information in the text:

1) [Questions about temporal order of events](#temporal_order)
2) [Questions about coreference](#coreference)
3) [Factual questions](#factual)

The following general tips apply to all of them.
 
## <a id="qtext_tips"/></a> What questions are easy for AI systems?

Despite the prevalence of AI virtual assistants such as Siri and Google Assistant, they actually understand very little of natural language. In particular, for the question answering task, it has been shown that multiple-choice questions can often be answered by simple text matching, without any language understanding at all. Consider the following example:

```xml
Text: John ate some cake. Fred ate a sandwich. Mary had a beer.
Question: What did John eat?
(a) a cake (correct)
(b) a sandwich
(c) a beer
```
  
The system does not actually need to know anything about eating or cakes to select the correct answer. It does not even need to know that beer is a drink and not food. It just needs to scan the text and select the answer option that is the closest to the question words "John" and "eat". 

Unfortunately, such shallow questions are the easiest for the humans to write, and they have been prevalent in the question answering datasets until now. Our goal for QuAIL is to have **both** shallow questions like the above, and questions that were made more difficult by paraphrasing or other strategies, so that in evaluating AI systems we can see what reasoning strategies they are capable of. 

Compare the above shallow question with a paraphrased version:

```xml
Text: John ate some cake. Fred ate a sandwich. Mary had a beer.
Question: What did John consume?
(a) a desert (correct)
(b) a sandwich
(c) a beer
```

This would be much more difficult to answer without language understanding. The question is paraphrased: the most important meaningful words in this question are "John" and "eat". Assuming that the text is long and John is mentioned elsewhere, the system will look for the mention of "eat". Since it is paraphrased as "consumed", the system will at least have to know that these are synonyms. The correct answer is also paraphrased: the system will need to know at least what type of food it is. 

Note that this strategy can also fail if only the correct option is always paraphrased: an AI system can simply learn to always find the option that is not present in the text directly. Theoretically there are 3 ways to handle this:

1. Paraphrasing the question and all three answer options (very effective, but the most time-consuming);

2. Since QuAIL has 2 questions of each type for each text, one could be paraphrased, and the other left unparaphrased, so that a system would be punished in 50% of cases if it learned to simply learn to look for the option that does not have a direct match in the text. 

We will refer to option (2) as 2/1 paraphrasing strategy: only the correct answer in one question, correct + one incorrect in another. If possible, the incorrect option that is mentioned the closest to the question words should the one left unparaphrased. Since our challenge set is small and only meant for testing, we did not do this consistently, aiming to make all questions as difficult as possible - but generally to develop a training dataset this is probably the way to go: the model should be taught that the text is a useful, but insufficient source of information. 
   
Below we describe extra strategies we used for specific question types.

## <a id="temporal_order"/></a> Temporal order questions

Temporal order questions are questions about the order of two events in the text. Two common forms of this question are:

* Questions for selecting the existing temporal relation to another event: 

```xml
Text: John ate some cake. He went to meet Fred. They watched some TV with Mary.
Question: When did John eat the cake?
(a) before meeting Fred (correct)
(b) after meeting Fred 
(c) while watching TV with Mary
```

* Questions for selecting an event that matches the given temporal relation:

```xml
Text: John ate some cake. He went to meet Fred. They watched some TV with Mary.
Question: After John met Fred:
(a) he watched TV (correct)
(b) he ate the cake
(c) he kissed Mary
```

In both of these examples the incorrect option (c) could be excluded without any language understanding, just by selecting the answers that have matching excerpts the closest to the words in the question. 

In addition to the direct-match problem, an AI system could arrive at the correct answer by simply assuming that the correct temporal order is the same as the order in which the events are mentioned in the text. In both examples above, the before/after relation that is being targeted is the same as the narrative order in the text.

To make these questions more difficult, paraphrasing could be performed as follows:


```xml
Text: John ate some cake. He went to meet Fred. They watched some TV with Mary.
Question: When did John have a sweet dish?
(a) before he encountered Fred (correct)
(b) after meeting Fred 
(c) while enjoying TV with Mary
```

Note that option (b) is left unparaphrased, because it would be more confusing: the word "meet" is mentioned closer to the information in the question. 

In some texts there is a faster way to make a question more difficult. If the text contains a mention of an intention, hope, wish, a conditional statement, an idiom, or some other event that did not actually happen, they make great distractors for a system that relies on the narrative order of events. If it is possible to find and change one of the incorrect answer options for such an distractor that is the same distance or closer to the question words than the correct answer, then it's not necessary to paraphrase.

```xml
Text: John went to meet his friend Fred. On the way, he thought he could kill for some cake. There was actually no desert, but they watched some TV with Mary.
Question: After John met Fred:
(a) he watched TV (correct)
(b) he killed somebody for a cake
(c) he kissed Mary
```

## <a id="coreference"/></a> Character identity

Questions about the identity of the character, entity or event ideally target a coreference relation, i.e. require the model to resolve who/what in the text is referred to by a pronoun (I/my/me, you/your/yours, he/his/him, she/her, it, we/us/ours, they/them/theirs).  For example:

```xml
Text: John went to meet his friend Fred. On the way, he thought he could kill for some cake. There was actually no desert, but they watched some TV with Mary.
Question: Who wanted to eat a cake?
(a) John (correct)
(b) Fred
(c) Mary
```

We can't do anything about the names, but such questions can be made more difficult by simply paraphrasing the question:

```xml
Text: John went to meet his friend Fred. On the way, he thought he could kill for some cake. There was actually no desert, but they watched some TV with Mary.
Question: Who wanted to eat something sweet?
(a) John (correct)
(b) Fred
(c) Mary
```
<a id="coref-relation"></a>
In many cases the Turkers misunderstood the task and instead asked for about the name, job, relation or another characteristic of a character:

```xml
Text: John went to meet his friend Fred. On the way, he thought he could kill for some cake. There was actually no desert, but they watched some TV with Mary.
Question: Who is Fred?
(a) John's buddy (correct)
(b) John's boss
(c) John's cousin
```
 
This is however good for us, because we also get a lot of such questions about characters and their relations in the ["unanswerable" category](#unanswerable). The fastest way to make these questions more difficult is to reuse the answers, treating them as paraphrases of somebody's name (John in this case), and to make the question target a coreferential mention of John: 
 
```xml
Text: John went to meet his friend Fred. On the way, he thought he could kill for some cake. There was actually no desert, but they watched some TV with Mary.
Question: ~~Who is Fred?~~ Who wanted to eat something sweet?
(a) Fred's friend (correct)
(b) Fred's boss
(c) Fred's cousin
``` 

Sometimes the Turker questions already require some extra reasoning beyond finding the nearest matching name, and we don't have to do anything about them. For example, in this case the system would have to know that hosts usually stay with their guests to respond correctly:

```xml
Text: John went to meet Fred. He thought he could kill for some cake. There was actually no desert, but they watched some TV with Mary.
Question: Who watched TV?
(a) John, Fred and Mary (correct)
(b) John and Mary
(c) Fred and Mary
```

# <a id="factual"/></a> Factual questions

Factual questions may target all other kinds of information that is available directly in the text and does not target temporal relations, coreference, or causal relations. Because there is no restriction by type of information, we might expect a lot of variety, but in fact Turkers tend to write simple questions based on the information in just one sentence. For example:

```xml
Text: John went home.
Question: Where did John go?  
(a) home (correct)
(b) conference
(c) Fred's office
```

Even if all the incorrect options are found in the text, it will be easy for an AI system to answer this question correctly simply by finding the answer that is mentioned the closest to "go"/"went" and "John". This of course does not actually require understanding language.

The fastest way to make such questions more difficult is to paraphrase the correct answer:

```xml
Text: John went home.
Question: Where did John go?  
(a) the place where he lives (correct)
(b) conference 
(c) Fred's office / "the place where Fred works" (if 2 options are paraphrased)
```

If there are any numbers, it is a good idea to spell them if they are digits in the text, and vice versa:

```xml
Text: John ate 2 cakes.
Question: How many desert portions did John have? (paraphrased from "How many cakes did John eat?")  
(a) two (correct) 
(b) three 
(c) one
``` 

# <a id="qworld"/></a> Questions requiring world knowledge

QuAIL has five types of such questions:

   * [Questions about causality](#causality)
   * [Questions about subsequent states](#subsequent_state)
   * [Questions about properties of characters and entities](#properties)
   * [Questions about event durations](#event_duration)
   * [Questions about belief states](#information_attribution)         

What they have in common is that the correct answer is not found in the text, but it should be easy to infer with world knowledge.

# <a id="qworld-tips"/></a> What kinds of world knowledge questions are easy for an AI system?

Since such questions do not have answers present in the text, an AI system has to make use of external knowledge to make the choice. So there is no need to paraphrase anything. However, we have an opposite problem: a system can simply scan the answers, find that they are not found in the text, and only from that it would figure out that it needs to use external knowledge resources. In other words, there is an obvious statistical difference between world knowledge questions and text questions that gives away a part of the game. 

The solution to this problem is the opposite of our 2/1 paraphrasing strategy: the 2/1 text mention strategy. In other words, the distractor answers for world knowledge questions *should* contain plausible words from the text. Ideally, if we have 2 questions of each type, one of them would have one such distractor, and the other one two (to make it look more like the text-based questions). 

For instance, the following causality question is not answerable from the text:

```xml
Text: The conference lasted for hours, everybody was anxious for a coffee break. John went out and ate a cake.
Question: Why did John consume a desert? 
(a) because he was hungry (correct)
(b) because he went to a conference
(c) because he was anxious
``` 

Note that the correct answer is not found in the text directly, but both incorrect options have content words that are found in the text ("anxious", "conference"). While such answer options are easy to reject for humans, they are in fact quite difficult for an AI system if it relies too much on naive text matches.

## <a id="causality"/></a> Questions about causal relations

Causal questions target causal relations between two events. This is in fact between text-based and world-knowledge-based questions, because the Turkers wrote both of them (although we asked specifically for the text-based questions):

* the correct answer (and sometimes the causal relation) are spelled out in the text: 

```xml
Text: John was hungry after the conference, so he ate a cake.
Question: Why did John eat a cake?  
(a) because he was hungry (correct) 
(b) because went to a conference
(c) because he wanted to eat something sweet
``` 

* the correct answer and the causal relation are not spelled out in the text, but are easy to infer with world knowledge:

```xml
Text: The conference lasted for hours, everybody was anxious for a coffee break. John went out and ate a cake.
Question: Why did John eat a cake?
(a) because he was hungry (correct) 
(b) because he went to a conference
(c) because he was anxious
``` 

If the correct answer is found in the text, we paraphrase it (and if the data were meant for training, we would only paraphrase half the cases):

```xml
Text: John was hungry after the conference, so he ate a cake.
Question: Why did John consume a desert?
(a) because he was starving (correct)
(b) because he went to a conference / "because he went to a research event"
(c) because he wanted to eat something sweet
``` 

In term of reasons provided, note that answer options (b) is easy to rule out for people: we know that conferences on their own do not induce hunger or happiness. But, if they are mentioned in the text close to the reason for something, they would likely confuse a naive AI system that relies on lexical cooccurrence too much. 

If the correct answer is *not* found in the text, we can use the opposite strategy with distractors-mentioned-in-text:

```xml
Text: The conference lasted for hours, everybody was anxious for a coffee break. John went out and ate a cake.
Question: Why did John consume a desert? 
(a) because he was hungry (correct) 
(b) because he went to a conference
(c) because he was anxious
``` 

## <a id="subsequent_state"/></a> Questions about subsequent states

Questions about subsequent states target the state of a character or an entity after some event. For example:

```xml
Text: John was walking all day.
Question: After the end of the story, what is probably true about John?
(a) John is tired (correct)
(b) John wants to do more walking
(c) John wants to see Mary
```

The same strategy for balancing distractors that are mentioned in the text applies. In this example we mention a distractor word "walking" in option (b), so in the other subsequent state question for this batch we need to mention a distractor word in both incorrect options. 

A common pattern for the Turkers is to ask about the continuation of the story, and people may have genuine differences about what is the most plausible answer. If several answer options look plausible, it is best to change the incorrect answers so as to make them look less plausible for a human reader.

## <a id="properties"/></a> Questions about properties of characters and entities

This category is for questions that target various properties and attributes of characters and entities that can be inferred on the basis of events in the text. For example: 

```xml
Text: John finished off the cake on his own.
Question: What is probably true about John?
(a) John has a sweet tooth (correct)
(b) John hates sweets
(c) John baked the cake before eating it
```

The ususal 2/1 mention rule applies: one incorrect option with a text mention in one question of this type, two in another. In this case, there is one such mention: event (c) is not in the text, and is unlikely (although not completely implausible), but it has text mentions ("cake", "eat"), which makes it a good distractor. 

## <a id="event_duration"/></a> Questions about event durations

Questions about event duration ask how long an event mentioned in the text probably lasted. For example:

```xml
Text: John ate a sandwich.
Question: How long did it probably take John to eat a sandwich?
(a) a few minutes (correct)
(b) a few hours
(c) a few days
```

This looks like a question requiring world knowledge, but an AI system that simply learned that "eat" occurs more frequently in the context of "minutes", rather than "hours" or "days", would still arrive at the correct answer without understanding anything about event duration. 

To make such a question more difficult, one strategy is to change the answer options so as to use the same unit of measurement:

```xml
Text: John ate a sandwich.
Question: How long did it probably take John to eat a sandwich?
(a) less than 15 minutes (correct)
(b) less than 1 minute
(c) less than 30 minutes
```

It is best to try to balance these questions so that the correct answer is not always the smallest, the middle or the largest value. It also helps to use as an incorrect option a number mentioned in the text, if any.

Another problem with some Turker questions is that they tend to use words like "a few, "several" for the correct answer, and numerical expressions for the incorrect ones, which gives it away. It should be the same time unit for all the answers.

## <a id="information_attribution"/></a> Belief states

These questions target belief states of characters: they may question what the characters feel or think about something/someone. 

```xml
Text: John had a sandwich and went to meet Fred. Fred was wearing green shoes and pink trousers.
Question: What does John probably think about Fred?
(a) Fred didn't choose his clothes well this morning (correct)
(b) Fred looked great
(c) Fred was looking for a sandwich.
```

Note that inclusion of "sandwich" in (c) makes the question extra challenging, as it is mentioned in the text close to a mention of Fred. 

# <a id="unanswerable"/></a> Unanswerable questions

The Turkers were asked to write these questions so that either all or none of the answer options would be present in the text. 

 * If all three options are present in the text, we can make the question more difficult by paraphrasing one of them (or two, if there are 2 such questions);

 * If none of the options are present in the text, we can change one of the incorrect options so that it mentions something from the text (or two, if there are 2 such questions per text).

From our observations, the Turkers mostly opted for the latter. 

A lot of these questions are about the identity of relation between characters. For example:

```xml
Text: John went to meet Fred. On the way, he thought he could kill for some cake. There was actually no desert, but they watched some TV with Mary.
Question: Who is Fred?
(a) John's friend 
(b) John's boss
(c) John's cousin
```

Because there are so many of them, it would be easy for an AI system to just learn to react to the "who" word or the answers with options about jobs/relations. However, we also have many such questions in the character identity category, so it should balance things out. 

One lesson from QuAIL for the future is that the unanswerable questions don't naturally get generated with much diversity: the Turkers just go for the easiest factoid question type. The instructions and prompts for unanswerable questions should be as diverse as for other question types.
