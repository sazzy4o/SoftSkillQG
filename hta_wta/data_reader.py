from posixpath import join
import random, ast
from typing import Tuple
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from datasets import load_dataset
from pandas.io.pytables import dropna_doc

# pd.set_option('max_colwidth', None)
random.seed(0)

class reader:
    """
    Output should have:
    'context', 'question', 'answer', 'skillName (for dream)'
    """

    def __init__(self, adding_answers) -> None:
        self.sep_token = " <sep> "
        self.ans_sep_token = " <ans> "
        self.question_with_answer = adding_answers

    def read(self, dataset_name:str='squad_v2') -> dict:
        """
        SQuAD: id, title, context, question, answers
            size: 142,192
            splits: train, validation
        COSMOS: id, context, question, answer[0,1,2,3], label
            size: 35,210
            splits: train, test, validation
        narrativeqa: document (id, kind, url, start, end, summary(text), text), question (text, tokens), answers[text, tokens]
            size: 46,765
            splits: train, test, validation
        RACE: example_id, article, question, answer, options
            size: 97,687
            splits: train, test, validation
        """
        self.dataset_name = dataset_name
        
        # load the training file
        func_mapper = {'squad_v2': self.process_squad, 'cosmos_qa': self.process_cosmos_qa, 
                        'narrativeqa': self.process_narrativeqa, 'dream': self.process_dream, 'dream_v3': self.process_dream_v3,
                        'boolq': self.process_boolq, 'tweet_qa': self.process_tweetq, 'quail': self.process_quail,
                        'DuoRC': self.process_duorc, 'MCTest': self.process_mctest, 'MCScript2.0': self.process_mcscript,
                        'WikiQA': self.process_wikiqa}
        train, validation, test = func_mapper[self.dataset_name]()
        return train, validation, test
    
    def split_by_ratio(self, data, split_on='skillName', frac=0.2):
        # print(data.shape)
        assert split_on in data.columns
        trains = []
        vals = []
        for item in data[split_on].unique():
            item_data = data[data[split_on] == item].reset_index(drop=True)
            val_ = item_data.sample(frac=frac, random_state=0)
            train_ = item_data[~item_data.index.isin(val_.index)].reset_index(drop=True)
            trains.append(train_)
            vals.append(val_)
        trains = pd.concat(trains, axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
        vals = pd.concat(vals, axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
        # print(trains.shape, vals.shape)
        return trains, vals

    def length_distribution(self, train, validation):
        print('length_distribution =========================')
        train['len'] = train['context'].str.split().apply(len)
        validation['len'] = validation['context'].str.split().apply(len)
        print('context: ')
        print('_____train: ', np.max(train['len']), np.mean(train['len']))
        print('validation: ', np.max(validation['len']), np.mean(validation['len']))
        print()
        train['len'] = train['question'].str.split().apply(len)
        validation['len'] = validation['question'].str.split().apply(len)
        print('question: ')
        print('_____train: ', np.max(train['len']), np.mean(train['len']))
        print('validation: ', np.max(validation['len']), np.mean(validation['len']))
        print('=============================================')

    def filter_length(self, data, based_on='context', text_length=450):
        data['length'] = data[based_on].map(lambda txt: len(txt.split()))
        data = data[data['length'] < text_length].reset_index(drop=True)
        del data['length']
        return data
    
    def process_squad(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dataset = load_dataset(self.dataset_name)
        train, test = dataset['train'], dataset['validation']
        
        def create_df(data, test=False, include_answers=False):
            df = pd.DataFrame({'context': data['context'], 'question': data['question'], 'answer': list(map(lambda item: item['text'][0] if len(item['text']) else '', data['answers']))})
            
            df['len'] = df['answer'].map(lambda ans: len(ans))
            df = df[(df['len'] > 0)].reset_index(drop=True)
            del df['len']
            return aggregate_questions(df, include_answers=include_answers)
        
        def aggregate_questions(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + col['answer'], axis=1)
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(x)), 'answer': list})
            return data
        
        train_tmp = create_df(train, include_answers=self.question_with_answer)
        test = create_df(test, test=True, include_answers=self.question_with_answer)

        train = train_tmp.sample(frac=0.9, random_state=0)
        validation = train_tmp.drop(train.index).reset_index(drop=True)
        train = train.reset_index(drop=True)
        return train, validation, test

    def process_cosmos_qa(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dataset = load_dataset(self.dataset_name)
        train, validation, test = dataset['train'], dataset['validation'], dataset['test']
        # print(test[10])
        # exit(1)
        def create_df(data, include_answers=False):
            df = pd.DataFrame({'context': data['context'], 'question': data['question'], 
                                'answer': map(lambda item: item[f'answer{item["label"]}'] if item["label"] != -1 else '', data)})
            df['len'] = df['answer'].map(lambda ans: len(ans))
            df = df[(df['len'] > 0)].reset_index(drop=True)
            del df['len']
            return aggregate_questions(df, include_answers=include_answers)
        
        def aggregate_questions(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + col['answer'], axis=1)
            
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(set(x))), 'answer': list})
            return data

        train = create_df(train, include_answers=self.question_with_answer)
        validation = create_df(validation, include_answers=self.question_with_answer)
        test = create_df(test, include_answers=self.question_with_answer)

        return train, validation, test

    def process_narrativeqa(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dataset = load_dataset(self.dataset_name)
        train, validation, test = dataset['train'], dataset['validation'], dataset['test']
        
        def create_df(data, include_answers=False):
            df = pd.DataFrame({'context': map(lambda item: item['summary']['text'], data['document']), 
                                'question': map(lambda item: item['text'], data['question']), 
                                'answer': map(lambda item: item[0]['text'], data['answers'])}
                                # 'answer': map(lambda item: list(set([item[i]['text'].title() for i in range(len(item))])), data['answers'])}
                                )
            return aggregate_questions(df, include_answers=include_answers)
        
        def aggregate_questions(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + col['answer'], axis=1)
            
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(x)), 'answer': list})
            return data

        train = create_df(train, include_answers=self.question_with_answer)
        validation = create_df(validation, include_answers=self.question_with_answer)
        test = create_df(test, include_answers=self.question_with_answer)
        
        # Filter the questions length
        train = self.filter_length(train, based_on='context', text_length=450)
        train['question'] = train['question'].map(lambda x: self.sep_token.join( x.split(self.sep_token)[:8] ))
        
        return train, validation, test

    def process_dream(self)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv("/home/bghanem/projects/EyeRead_QA/data/tmp_csv/questions_classification.csv")
        data_skills = pd.read_csv("/home/bghanem/projects/EyeRead_QA/data/dream-data-v2/skills_edited.csv")
        # Fixing for the skills csv file from lauren
        data_skills['Subskill Category'] = data_skills.apply(lambda row: 
                    row['Subskill Category'] if not pd.isna(row['Subskill Category']) else row['name'], axis=1)
        data_skills['name'] = data_skills['Subskill Category']
        data_skills = data_skills[['id', 'name']]
        data_skills.rename(columns={'id': 'skillId_skills', 'name': 'skillName'}, inplace=True)
        data = pd.merge(data, data_skills, how='left', left_on='skillId', right_on='skillId_skills')
        del data['skillId_skills']
        
        data = data[(~data['question'].str.contains('____')) & (~data['question'].str.contains('order'))]
        data = data.drop_duplicates('question').reset_index(drop=True)

        # Taking just the first N skills (majority classes)
        skills_filter = data.groupby('skillName').size().reset_index(drop=False).rename(columns={0: 'count'})
        skills_filter = skills_filter.sort_values('count', ascending=False).reset_index(drop=True)
        skills_filter = skills_filter.iloc[:9, :] # 9...5
        # print(skills_filter)
        # exit(1)
        
        
        data = data[data['skillName'].isin(skills_filter['skillName'].unique())].reset_index(drop=True)
        # print(data[['question','skillName', 'Question Type']].groupby(by=['skillName', 'Question Type']).count())
        # exit(1)
        data = data[['text', 'question', 'correctAnswers', 'skillName']].rename(
                                    columns={'text': 'context', 'correctAnswers': 'answer'})

        # data = data.sort_values('skillName', ascending=False).reset_index(drop=True)
        # for sn in data['skillName'].unique().tolist():
        #     print(sn)
        #     tmp = data[data['skillName'] == sn]
        #     print(len(tmp['context'].unique()))
        #     print(len(tmp['question']))

        #     tmp['con_word'] = tmp['context'].map(lambda x: len(x.split(' ')))
        #     tmp['ques_word'] = tmp['question'].map(lambda x: len(x.split(' ')))
        #     tmp['ans_word'] = tmp['answer'].map(lambda x: len(ast.literal_eval(x)[0].split(' ')))

        #     print(average(tmp['con_word']), max(tmp['con_word']))
        #     print(average(tmp['ques_word']), max(tmp['ques_word']))
        #     print(average(tmp['ans_word']), max(tmp['ans_word']))
        #     print()
        # exit(1)

        def aggregate_answers(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + ast.literal_eval(col['answer'])[0], axis=1)
            return data
        
        data = aggregate_answers(data, include_answers=self.question_with_answer)
        data['context'] = data.apply(lambda row: row['skillName'] + ' </s> ' + row['context'], axis=1)
        tr, val = self.split_by_ratio(data, split_on='skillName', frac=0.12)
        tr, tst = self.split_by_ratio(tr, split_on='skillName', frac=0.3)
        print(tr.shape, val.shape, tst.shape)

        # exit(1)
        return tr, val, tst

    def process_boolq(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dataset = load_dataset(self.dataset_name)
        train, validation = dataset['train'], dataset['validation']
        # print(validation[111])
        # exit(1)
        def create_df(data, include_answers=False):
            df = pd.DataFrame({'context': data['passage'], 'question': data['question'], 'answer': data['answer']})
            df['question'] = df['question'].map(lambda x: x+'?' if x[-1] != '?' else x)
            return aggregate_questions(df, include_answers=include_answers)
        
        def aggregate_questions(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + str(col['answer']), axis=1)
            
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(set(x))), 'answer': list})
            return data

        train = create_df(train, include_answers=self.question_with_answer)
        validation = create_df(validation, include_answers=self.question_with_answer)
        
        # Filter long documents
        train = train[train.context.str.split().apply(len) < 512].reset_index(drop=True)
        train['question'] = train['question'].map(lambda x: self.sep_token.join(x.split(self.sep_token)[:11]))

        return train, validation, pd.DataFrame({'context': [], 'question': [], 'answer': []})

    def process_tweetq(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_json(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/train.json')
        validation = pd.read_json(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/dev.json')
        test = pd.read_json(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/test.json')
        
        def create_df(data, include_answers=False):
            df = pd.DataFrame({'context': data['Tweet'], 'question': data['Question'], 
                                'answer': data['Answer']})
            return aggregate_questions(df, include_answers=include_answers)
        
        def aggregate_questions(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + col['answer'][0], axis=1)
            
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(set(x))), 'answer': list})
            return data

        train = create_df(train, include_answers=self.question_with_answer)
        validation = create_df(validation, include_answers=self.question_with_answer)
        return train, validation, pd.DataFrame({'context': [], 'question': [], 'answer': []})

    def process_quail(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dataset = load_dataset(self.dataset_name)
        train, validation = dataset['train'], dataset['validation']
        # print(train[1])
        # exit(1)
        def create_df(data, include_answers=False):
            df = pd.DataFrame({'context': data['context'], 'question': data['question'], 'answer': data['answers'], 
                                'correct_answer_id': data['correct_answer_id']})
            df['answer'] = df.apply(lambda row: row['answer'][row['correct_answer_id']], axis=1)
            del df['correct_answer_id']
            return aggregate_questions(df, include_answers=include_answers)
        
        def aggregate_questions(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + col['answer'], axis=1)
            
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(set(x))), 'answer': list})
            return data

        train = create_df(train, include_answers=self.question_with_answer)
        validation = create_df(validation, include_answers=self.question_with_answer)
        
        # Filter long documents
        train['question'] = train['question'].map(lambda x: self.sep_token.join(x.split(self.sep_token)[:7]))
        validation['question'] = validation['question'].map(lambda x: self.sep_token.join(x.split(self.sep_token)[:7]))

        return train, validation, pd.DataFrame({'context': [], 'question': [], 'answer': []})

    def process_duorc(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_json(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/train.json')
        validation = pd.read_json(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/dev.json')
        test = pd.read_json(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/test.json')
        
        def create_df(data, include_answers=False):
            if include_answers:
                data['question'] = data['qa'].map(lambda x:  self.sep_token.join([ 
                                itm['question'] + self.ans_sep_token + itm['answers'][0]
                                            for itm in x if itm['no_answer'] == False]))
            else:
                data['question'] = data['qa'].map(lambda x:  self.sep_token.join([ itm['question'] for itm in x ]))
            data['answer'] = data['qa'].map(lambda x:  [ itm['answers'][0] for itm in x if itm['no_answer'] == False])
            
            del data['title'], data['qa'], data['id']
            data.rename(columns={'plot': 'context'}, inplace=True)
            return data

        train = create_df(train, include_answers=self.question_with_answer)
        validation = create_df(validation, include_answers=self.question_with_answer)
        test = create_df(test, include_answers=self.question_with_answer)
        return train, validation, test

    def process_mctest(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        header = ['Id', 'properties', 'story', 'q1', 'q1-1', 'q1-2', 'q1-3', 'q1-4', 
                            'q2', 'q2-1', 'q2-2', 'q2-3', 'q2-4', 'q3', 'q3-1', 'q3-2', 'q3-3', 'q3-4', 
                            'q4', 'q4-1', 'q4-2', 'q4-3', 'q4-4']
        train = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/train.tsv', sep='\t', names=header)
        validation = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/dev.tsv', sep='\t', names=header)
        
        train_labels = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/train.ans', sep='\t', names=[1, 2, 3, 4])
        validation_labels = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/dev.ans', sep='\t', names=[1, 2, 3, 4])
        mapper = {'A': 1, 'B': 2, 'C': 3, 'D':4}
        train_labels = train_labels.applymap(lambda x: mapper[x])
        validation_labels = validation_labels.applymap(lambda x: mapper[x])

        train = pd.concat([train, train_labels], axis=1)
        validation = pd.concat([validation, validation_labels], axis=1)

        def merging_columns(data, include_answers=False):
            data['question'] = data.apply(lambda row: self.sep_token.join( [row['q1'], row['q2'], row['q3'], row['q4']] ), axis=1)
            data['answer'] = data.apply(lambda row: [row[f'q1-{row[1]}'], row[f'q2-{row[2]}'], row[f'q3-{row[3]}'], row[f'q4-{row[4]}']], axis=1)
            
            data = data[['question', 'story', 'answer']]
            data['question'] = data['question'].map(lambda txt: txt.replace('multiple: ', '').replace('one: ', ''))
            data.rename(columns={'story': 'context'}, inplace=True)
            data = data[~data['question'].str.contains("\.\.\.")].reset_index(drop=True)
            if include_answers:
                data['question'] = data['question'].map(lambda q: q.split(self.sep_token))
                data['question'] = data.apply(lambda col: self.sep_token.join(
                                                self.ans_sep_token.join(q_a_pain) for q_a_pain in 
                                                    zip(col['question'], col['answer'])), axis=1)
            return data

        train = merging_columns(train, include_answers=self.question_with_answer)
        validation = merging_columns(validation, include_answers=self.question_with_answer)
        return train, validation, pd.DataFrame({'context': [], 'question': [], 'answer': []})

    def process_mcscript(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        def read_xml(split):
            data = []
            import xml.etree.ElementTree as ET
            root = ET.parse(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/{split}.xml').getroot()
            for instance in root.findall('instance'):
                for content_tags in instance:
                    if content_tags.tag == 'text':
                        text = content_tags.text
                    else:
                        for questions in content_tags:
                            question_content = questions.attrib['text']
                            for answers in questions:
                                if answers.attrib['correct'] == "True":
                                    answer = answers.attrib['text']
                            data.append([text, question_content, answer])
            return pd.DataFrame(data, columns=['context', 'question', 'answer'])

        train = read_xml('train')
        validation = read_xml('dev')
        test = read_xml('test')

        def aggregate_questions(data, include_answers=False):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + col['answer'], axis=1)
            
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(set(x))), 'answer': list})
            data = self.filter_length(data, based_on='context', text_length=512)
            data['question'] = data['question'].map(lambda x: self.sep_token.join(x.split(self.sep_token)[:9]))
            return data

        train = aggregate_questions(train, include_answers=self.question_with_answer)
        validation = aggregate_questions(validation, include_answers=self.question_with_answer)
        test = aggregate_questions(test, include_answers=self.question_with_answer)
        return train, validation, test

    def process_wikiqa(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/train.tsv', sep='\t')
        validation = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/dev.tsv', sep='\t')
        test = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/data/qa_datasets/{self.dataset_name}/test.tsv', sep='\t')

        def aggregate_questions(data):
            data = data.groupby(['Question'], as_index=False).agg(
                    {'Sentence': lambda x: " ".join(x), 'Label': lambda x: " ".join([f'{itm}' for itm in x])})
            data = data[data['Label'].str.contains('1')].rename(columns={'Question': 'question', 'Sentence': 'context'}).reset_index(drop=True)
            data['answer'] = 'no answers'
            del data['Label']
            
            data['question'] = data['question'].map(lambda x: x.lower()+'?'.lower() if x[-1] != '?' else x.lower())
            data = data.groupby('context', as_index=False).agg(
                    {'question': lambda x: self.sep_token.join(list(set(x))), 'answer': list})
            return data
        
        train = aggregate_questions(train)
        validation = aggregate_questions(validation)
        test = aggregate_questions(test)

        return train, validation, test

    def process_dream_v3(self)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv("/home/bghanem/projects/EyeRead_QA/data/tmp_csv/questions_classification.csv")
        data_skills = pd.read_csv("/home/bghanem/projects/EyeRead_QA/data/dream-data-v2/skills_edited.csv")
        
        # Fixing for the skills csv file from lauren
        data_skills['Subskill Category'] = data_skills.apply(lambda row: 
                    row['Subskill Category'] if not pd.isna(row['Subskill Category']) else row['name'], axis=1)
        data_skills['name'] = data_skills['Subskill Category']
        data_skills = data_skills[['id', 'name']]
        data_skills.rename(columns={'id': 'skillId_skills', 'name': 'skillName'}, inplace=True)
        data = pd.merge(data, data_skills, how='left', left_on='skillId', right_on='skillId_skills')
        del data['skillId_skills']
        data = data[(~data['question'].str.contains('____')) & (~data['question'].str.contains('order'))]

        # Taking just the first N skills (majority classes)
        skills_filter = data.groupby('skillName').size().reset_index(drop=False).rename(columns={0: 'count'})
        skills_filter = skills_filter.sort_values('count', ascending=False).reset_index(drop=True)
        skills_filter = skills_filter.iloc[:16, :]
        
        data = data[data['skillName'].isin(skills_filter['skillName'].unique())].reset_index(drop=True)
        # Counting how many inferencial and extractive questions are there
        # print(data[['question','skillName', 'Question Type']].groupby(by=['skillName', 'Question Type']).count())
        # exit(1)
        data = data[['text', 'question', 'correctAnswers', 'skillName', 'potentialAnswers']].rename(
                                    columns={'text': 'context', 'correctAnswers': 'answer', 'potentialAnswers': 'options'})
        # Remove the answer from the options
        data['options'] = data.apply(lambda x: list(set(eval(x['options'])) - set(eval(x['answer']))), axis=1)
        def aggregate_answers(data, include_answers):
            if include_answers:
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + ast.literal_eval(col['answer'])[0], axis=1)
                data['question'] = data.apply(lambda col: col['question'] + self.ans_sep_token + f" {self.ans_sep_token} ".join(col['options']), axis=1)
            return data
        
        data = aggregate_answers(data, include_answers=self.question_with_answer)
        data['context'] = data.apply(lambda row: row['skillName'] + ' </s> ' + row['context'], axis=1)
        tr, val = self.split_by_ratio(data, split_on='skillName', frac=0.12)
        tr, tst = self.split_by_ratio(tr, split_on='skillName', frac=0.3)
        print(tr.shape, val.shape, tst.shape)
        # print(tr[['context', 'answer', 'options']])
        # exit(1)
        return tr, val, tst


if __name__ == '__main__':
    x = reader(adding_answers=True)

    datasets = ['dream_v3']#, 'dream', 'squad_v2', 'cosmos_qa', 'narrativeqa', 'boolq', 'tweet_qa', 'quail', 'DuoRC', 'MCTest', 'MCScript2.0'] # 'WikiQA',
    factual = ['squad_v2', 'narrativeqa', 'boolq', 'tweet_qa', 'quail', 'MCTest']
    reasoning = ['cosmos_qa', 'DuoRC', 'MCScript2.0']

    trains = []
    valids = []
    tests = []
    for dataset in datasets:
        tr1, vd1, ts1 = x.read(dataset)
        trains.append(tr1)
        valids.append(vd1)
        if dataset == 'squad_v2' or dataset == 'dream' or dataset == 'dream_v3':
            tests.append(ts1)
        print(dataset, ': ', tr1.shape, vd1.shape, ts1.shape)
    tr = pd.concat(trains, axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    vd = pd.concat(valids, axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    ts = pd.concat(tests, axis=0).reset_index(drop=True)
    print('Datasets size: ', tr.shape, vd.shape, ts.shape)
    # tr.to_csv('/home/bghanem/projects/EyeRead_QA/data/tmp_csv/training_dream_withAnswersOptions_v3.csv', header=True, index=False)
    # vd.to_csv('/home/bghanem/projects/EyeRead_QA/data/tmp_csv/validation_dream_withAnswersOptions_v3.csv', header=True, index=False)
    # ts.to_csv('/home/bghanem/projects/EyeRead_QA/data/tmp_csv/test_dream_withAnswersOptions_v3.csv', header=True, index=False)
    # print('data saved..')
    exit(1)
    data = pd.concat([tr4, vd4, ts4], axis=0).reset_index(drop=True)
    random_idx = [random.randint(0, data.shape[0]-1) for idx in range(250)]
    # data = data.sort_values('context').reset_index(drop=True)
    data = data.iloc[random_idx, :].reset_index(drop=True)
    for i, row in data.iterrows():
        print(i)
        print('\n', row['context'].replace('\n', ' ').replace('\r', ' ').replace('\s{2,}', ' ')) #.split(' <sep> ')[1]
        print('question: ', row['question'])
        print('answer: #', row['answer'], '#')
        print('\n\n')