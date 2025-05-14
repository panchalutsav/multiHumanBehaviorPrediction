import pandas as pd 
import re
import numpy as np
import Levenshtein
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_MODEL = BertModel.from_pretrained('bert-base-uncased')
BERT_MODEL.eval()  # Set the model to evaluation mode

class Evaluation: # currenly works with two chars, will be updated for more characters
    def __init__(self, csvpath):
        self.df = pd.read_csv(csvpath) 
        self.predictions = self.df['prediction'] if 'prediction' in self.df.columns else self.df['output_episode_1']
        self.gtoutput = self.df['gtoutput'] 
    
    # convert the whole string of prediction into list of pairs
    def getpairs(self, x): 
        return re.findall(r'\(.*?\)\s\(.*?\)', x) # if there are two character 
    
    # "(char0, action0, object0),(char1, action1, object1)" -> ["(char0, action0, object0)", "(char1, action1, object1)"]  
    def getUniqueActions(self, x):
        return list(re.findall(r'\(.*?\)', x))
    
    def getActionObject(self, s):
        if s:
            parts = s.strip('()').split(',')
            if len(parts) >= 2:
                return parts[1].strip(), parts[2].strip()
        
    
    def gettimedaccuracies(self): # max upto 3 chars
        char0total, char1total, char2total = 0, 0, 0
        completeaccTotal = 0
        maxchars = 0
        for i in range(len(self.df)):
            preds = self.df.iloc[i]['prediction']
            gts = self.df.iloc[i]['gtoutput']
            predslist = self.getpairs(preds)
            gtlist = self.getpairs(gts)

            char0acc, char1acc, char2acc, completeacc = 0, 0, 0, 0
            
            for (preditem, gtitem) in zip(predslist, gtlist):
                predactions = self.getUniqueActions(preditem)  # [char0action, char1action]
                gtactions = self.getUniqueActions(gtitem) # [char0action, char1action]
                
                predactions.extend([" "] * (len(gtactions) - len(predactions))) # to make the length of predactions and gtactions same

                if predactions[0] == gtactions[0]:
                   char0acc += 1
                 
                if predactions[1] == gtactions[1]:
                    char1acc += 1
                
                if predactions[0] == gtactions[0] and predactions[1] == gtactions[1]:
                   completeacc += 1

                if len(gtactions) == 3:
                    if predactions[2] == gtactions[2]:
                        char2acc += 1

                if len(gtactions) > maxchars:
                    maxchars = len(gtactions)
                                
            char0total += char0acc/len(gtlist) 
            char1total += char1acc/len(gtlist)
            char2total += char2acc/len(gtlist)
            completeaccTotal += completeacc/len(gtlist)

        returnvalue = {
            "char0": round(char0total/len(self.df), 2),
            "char1": round(char1total/len(self.df), 2),
            "char2": round(char2total/len(self.df), 2),
            # "complete": round(completeaccTotal/len(self.df), 2)
        } if maxchars == 3 else {
            "char0": round(char0total/len(self.df), 2),
            "char1": round(char1total/len(self.df), 2),
            "complete": round(completeaccTotal/len(self.df), 2)
        }

        return returnvalue
    
    def get_bert_character_embeddings(self, text_list: list):
        """
        Get character-level embeddings using BERT for a list of strings.
        :param text_list: List of strings
        :return: Matrix of embeddings where each row corresponds to a string
        """
        all_embeddings = []
        for text in text_list:
            encoded_input = BERT_TOKENIZER(
                text, return_tensors='pt', padding=True, truncation=True)

            with torch.no_grad():
                output = BERT_MODEL(**encoded_input)

            last_hidden_state = output.last_hidden_state
            if last_hidden_state.shape[1] > 2:
                mean_embedding = torch.mean(last_hidden_state[:, 1:-1, :], dim=1)
            else:
                mean_embedding = torch.mean(last_hidden_state, dim=1)

            all_embeddings.append(mean_embedding.numpy().flatten())

        return np.array(all_embeddings)


    def compute_cosine_similarity(self, prediction: list, ground_truth: list) -> float:
        """
        Compute the cosine similarity between predicted interaction labels and ground truth labels using BERT embeddings.
        This helps capture semantic similarity between phrases.

        :param prediction: List of predicted interaction labels.
        :param ground_truth: List of ground truth interaction labels.
        :return: Cosine similarity score (0 to 1, where 1 means identical meaning)
        """
        if len(prediction) == 0:
            return 0

        pred_embeddings = self.get_bert_character_embeddings(prediction)
        gt_embeddings = self.get_bert_character_embeddings(ground_truth)

        mean_pred_embedding = np.mean(pred_embeddings, axis=0).reshape(1, -1)
        mean_gt_embedding = np.mean(gt_embeddings, axis=0).reshape(1, -1)

        return cosine_similarity(mean_pred_embedding, mean_gt_embedding)[0][0]
    
    def getcosinesimilarity(self):
        allcos = []
        for i in range(len(self.df)):
            preds = self.df.iloc[i]['prediction']
            gts = self.df.iloc[i]['gtoutput']
            prediction_list = self.getUniqueActions(preds)
            ground_truth_list = self.getUniqueActions(gts)
            prediction_list.extend([" "] * (len(ground_truth_list) - len(prediction_list)))
            cosine_similarity_score = self.compute_cosine_similarity(
            prediction_list, ground_truth_list)
            allcos.append(cosine_similarity_score)
        
        return np.mean(allcos)

    def getEditDistance(self):
        edscores = []
        for i in range(len(self.df)):
            pred, gt = self.df.iloc[i]['prediction'], self.df.iloc[i]['gtoutput']
            edit_distance = Levenshtein.distance(pred, gt) / max(len(pred), len(gt))
            edscores.append(edit_distance)
        return round(1 - np.mean(edscores), 2).item()
    
    def combine_actions_by_character(self, items):
        actions_by_character = {}
        for item in items:
            # Extract the character name (first item in the tuple)
            charname = item.split(',')[0].strip('() ')
            if charname not in actions_by_character:
                actions_by_character[charname] = []
            actions_by_character[charname].append(item)
        return actions_by_character

    def extract_verb_noun(self, action): # action is (charid, verb, noun)
        parts = action.strip('()').split(',')
        # print(parts)
        return parts[1], parts[2]

    def calculate_eds(self, groundtruth, pred):
        results = {}
        
        for key in groundtruth:
            gt_actions = groundtruth[key]
            pred_actions = pred.get(key, [])

            # Calculate accuracy
            correct = sum(1 for gt, pr in zip(gt_actions, pred_actions) if gt == pr)
            accuracy = correct / len(gt_actions) if gt_actions else 0

            # Calculate edit distance
            edit_distance = sum((Levenshtein.distance(gt, pr) / max(len(gt), len(pr))) for gt, pr in zip(gt_actions, pred_actions)) / len(gt_actions) if gt_actions else 0

            verb_edit_distance = 0
            noun_edit_distance = 0
            verbacc = 0
            nounacc = 0

            for gt, pr in zip(gt_actions, pred_actions):
                gt_verb, gt_noun = self.extract_verb_noun(gt)
                pr_verb, pr_noun = self.extract_verb_noun(pr)

                if gt_verb == pr_verb:
                    verbacc += 1
                
                if gt_noun == pr_noun:
                    nounacc += 1
                
                verb_edit_distance += (Levenshtein.distance(gt_verb, pr_verb) / max(len(gt_verb), len(pr_verb)))
                noun_edit_distance += (Levenshtein.distance(gt_noun, pr_noun) / max(len(gt_noun), len(pr_noun)))
 
            verb_edit_distance /= len(gt_actions) if gt_actions else 0
            noun_edit_distance /= len(gt_actions) if gt_actions else 0
            verbacc /= len(gt_actions) if gt_actions else 0
            nounacc /= len(gt_actions) if gt_actions else 0
                 
            results[key] = {
                'accuracy': round(accuracy, 2),
                'seq_ed': round(edit_distance, 2),
                'verb_ed': round(verb_edit_distance, 2),
                'noun_ed': round(noun_edit_distance, 2), 
                'verbacc': round(verbacc, 2),
                'nounacc': round(nounacc, 2),
            }
        return results
    

    def get_eds(self):
        df_results = []
        for i in range(len(self.df)):
            preds = self.df.iloc[i]['prediction']
            gts = self.df.iloc[i]['gtoutput']
            preds_unique = self.getUniqueActions(preds)
            gts_unique = self.getUniqueActions(gts)
            preds_sep_char = self.combine_actions_by_character(preds_unique) 
            gts_sep_char = self.combine_actions_by_character(gts_unique) 
            rowresult = self.calculate_eds(gts_sep_char, preds_sep_char) 
            rowresult['complete'] = {
                'accuracy': round(sum(rowresult[key]['accuracy'] for key in rowresult) / len(rowresult), 2),
                'seq_ed': round(sum(rowresult[key]['seq_ed'] for key in rowresult) / len(rowresult), 2),
                'verb_ed': round(sum(rowresult[key]['verb_ed'] for key in rowresult) / len(rowresult), 2),
                'noun_ed': round(sum(rowresult[key]['noun_ed'] for key in rowresult) / len(rowresult), 2), 
                'verbacc': round(sum(rowresult[key]['verbacc'] for key in rowresult) / len(rowresult), 2),
                'nounacc': round(sum(rowresult[key]['nounacc'] for key in rowresult) / len(rowresult), 2),
            }
            df_results.append(rowresult)
        
        # sum the individual metrics 
        df_verbacc = sum(result['complete']['verbacc'] for result in df_results)/ len(self.df)
        df_nounacc = sum(result['complete']['nounacc'] for result in df_results)/ len(self.df)
        df_verb_ed = sum(result['complete']['verb_ed'] for result in df_results)/ len(self.df)
        df_noun_ed = sum(result['complete']['noun_ed'] for result in df_results)/ len(self.df)
        df_seq_ed = sum(result['complete']['seq_ed'] for result in df_results)/ len(self.df)
        df_acc = sum(result['complete']['accuracy'] for result in df_results)/ len(self.df)
        df_char0_ed = sum(result['female1']['seq_ed'] for result in df_results) / len(self.df)
        # df_char1_ed = sum(result['female2']['seq_ed'] for result in df_results) / len(self.df)
        df_char2_ed = sum(result['male1']['seq_ed'] for result in df_results if 'male1' in result) / len(self.df)

        return {
            'verbacc': round(df_verbacc, 2),
            'nounacc': round(df_nounacc, 2),
            'char0_ed': round(df_char0_ed, 2), 
            # 'char1_ed': round(1-df_char1_ed, 2), 
            'char2_ed': round(df_char2_ed, 2), 
            'verb_ed': round(df_verb_ed, 2), 
            'noun_ed': round(df_noun_ed, 2), 
            'action_ed': round(df_seq_ed, 2), 
            'accuracy': round(df_acc, 2), 
        }

    def evaluate(self):
        timedaccuracies = self.gettimedaccuracies()
        cossim = self.getcosinesimilarity()
        # ed = self.getEditDistance()
        eds = self.get_eds()
        return {
            "timedaccuracies": timedaccuracies,
            'verbacc': eds['verbacc'],
            'nounacc': eds['nounacc'],
            "cosine_similarity": cossim, 
            "verb_ed": eds['verb_ed'],
            "noun_ed": eds['noun_ed'],
            "action_ed": eds['action_ed'],
            "char0_ed": eds['char0_ed'],
            # "char1_ed": eds['char1_ed'],
            "char2_ed": eds['char2_ed'],
            "accuracy": eds['accuracy'],
        }
    


if __name__ == "__main__":
    filename = "/home/pau1rng/utsav/codes/multi-human-behavior-prediction/ftoutputs/qwen72b-kitchen2.csv"
    ev = Evaluation(filename)
    print(ev.evaluate())


