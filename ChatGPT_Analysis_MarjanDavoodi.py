


'''
Goal: Project diagnosis-label pairs on moral dimensions, using embeddings from the GPT-3 model
Date created: 6.6.2023
Date last modified: 1.2.2024
#########  Marjan Davoodi ###########
'''

###################################

#Import libraries:

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import word_lists #Compiles morality anchor words in dictionarities with pos and neg keys. 

###################################

#Access OpenAI API:
openai.api_key= "OPENAI_API_KEY"


# Read csv file that compiles a list of words representing diagnosis-label pairs and keywords for the five moral dimensions:
# (Source of moral dimensions anchor words: https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic)

words = pd.read_csv(r'~/ChatGPT/words.csv')


# Function to extract word embeddings for the list of words:

def get_embedding_with_na(word, engine='text-embedding-ada-002'):
    try:
        embedding = get_embedding(word, engine=engine)
        return embedding
    except KeyError:
        return "NA"


#Add embedding column for the words:
words['embedding'] = words['word'].apply(lambda x: get_embedding_with_na(x, engine='text-embedding-ada-002'))


# Save it to a csv file:
words.to_csv('word_embeddings_na.csv', index=False, na_rep='NA')



#Read the word embeddings csv file:
csv_embeddings = pd.read_csv('word_embeddings_na.csv')



class dimension_lexicon_builtin:
    def __init__(self, direction_of_interest, csv_embedding):
        """
        Initialize an instance of the dimension_lexicon_builtin class.

        Parameters:
        - direction_of_interest (str): Specifies the dimension of interest ('purity', 'harm', 'ingroup', 'fairness', or 'authority').
        - csv_embedding (DataFrame): DataFrame containing embeddings.
        """
        self.direction_of_interest = direction_of_interest
        self.csv_embedding = csv_embedding

        # Set labels and keyword lists based on the specified dimension of interest
        if self.direction_of_interest == 'purity':
            self.pos_label = "pure"
            self.neg_label = "impure"
            self.all_pos = word_lists.purity['pos']
            self.all_neg = word_lists.purity['neg']
        elif self.direction_of_interest == 'harm':
            self.pos_label = "harm virtue"
            self.neg_label = "harm vice"
            self.all_pos = word_lists.harm['pos']
            self.all_neg = word_lists.harm['neg']
        elif self.direction_of_interest == 'ingroup':
            self.pos_label = "Ingroup virtue"
            self.neg_label = "Ingroup vice"
            self.all_pos = word_lists.ingroup['pos']
            self.all_neg = word_lists.ingroup['neg']
        elif self.direction_of_interest == 'fairness':
            self.pos_label = "fairness virtue"
            self.neg_label = "fairness vice"
            self.all_pos = word_lists.fairness['pos']
            self.all_neg = word_lists.fairness['neg']
        elif self.direction_of_interest == 'authority':
            self.pos_label = "authority virtue"
            self.neg_label = "authority vice"
            self.all_pos = word_lists.authority['pos']
            self.all_neg = word_lists.authority['neg']




# Create instances of the dimension_lexicon_builtin class for different moral dimensions:

semantic_direction_purity = dimension_lexicon_builtin('purity', csv_embedding) 
semantic_direction_harm = dimension_lexicon_builtin('harm', csv_embedding) 
semantic_direction_ingroup = dimension_lexicon_builtin('ingroup', csv_embedding) 
semantic_direction_fairness = dimension_lexicon_builtin('fairness', csv_embedding) 
semantic_direction_authority = dimension_lexicon_builtin('authority', csv_embedding) 



def calc_wordlist_mean_csv(wordlist, embedding_file):
    """
    Calculate the mean vector for a given wordlist using embeddings from an embedding_file.

    Parameters:
    - wordlist (list): List of words for which to calculate the mean vector.
    - embedding_file (DataFrame): DataFrame containing word embeddings.

    Returns:
    - meanvec (numpy array): Mean vector calculated from the word embeddings of the specified wordlist.
    """
    # Extract word vectors from the embedding file for the specified wordlist
    word_vectors = [embedding_file.loc[embedding_file['word'] == word].values[0][1:] for word in wordlist]

    # Convert to correct array format
    vector_arrays = [np.array(ast.literal_eval(arr[1])) for arr in word_vectors]

    # Calculate the mean vector
    meanvec = np.mean(vector_arrays, 0)

    # Normalize
    meanvec = preprocessing.normalize(meanvec.reshape(1, -1), norm='l2')

    return meanvec



class dimension:
    def __init__(self, semantic_direction, method='larsen'):
        """
        Initialize an instance of the dimension class.

        Parameters:
        - semantic_direction: An instance of the dimension_lexicon_builtin class.
        - method (str): Method for dimension extraction. Default is 'larsen'(Source: https://arxiv.org/abs/1512.09300)
        """
        self.semantic_direction = semantic_direction
        self.method = method
        self.csv_embedding = semantic_direction.csv_embedding

    def calc_dim_larsen_csv(self):
        """
        Calculate dimension vector using the Larsen method with CSV embeddings.

        Returns:
        - diffvec (numpy array): Dimension vector according to the Larsen method.
        """
        diffvec = calc_wordlist_mean_csv(self.semantic_direction.all_pos, self.csv_embedding) - 
                  calc_wordlist_mean_csv(self.semantic_direction.all_neg, self.csv_embedding)
        diffvec = preprocessing.normalize(diffvec.reshape(1, -1), norm='l2')
        return diffvec

    def cos_sim(self, inputwords):
        """
        Calculate cosine similarity between the dimension vector and input words.

        Parameters:
        - inputwords (list): List of words for cosine similarity calculation.

        Returns:
        - List of tuples: Word and corresponding cosine similarity.
        """
        assert type(inputwords) == list, "Enter word(s) as a list, e.g., ['word']"
        interesting_dim = self.dimensionvec().reshape(1, -1)
        cossims = []
        for i in np.array(inputwords):
            try:
                cossims.append(cosine_similarity(np.array(inputwords_vectors([i], self.csv_embedding)),
                                                     interesting_dim)[0][0])
            except KeyError:
                cossims.append(np.nan)
                continue
        return list(zip(inputwords, cossims))



#Diagnosis and label word lists:
diagnosis = ['schizophrenia', 'autism', 'alcoholism', 'addiction', 'psychopathy', 'narcissism'] 

label = ['schizophrenic', 'autistic', 'alcoholic', 'addict', 'psychopath', 'narcissist']



#Create an instance of the 'dimension' class for the purity dimension using the Larsen method:
larsen_purity_dir = dimension(semantic_direction_purity)

# Calculate cosine similarity between the purity dimension vector and diagnosis keywords
purity_diagnosis = larsen_purity_dir.cos_sim(diagnosis)

# Calculate cosine similarity between the purity dimension vector and label keywords
purity_label = larsen_purity_dir.cos_sim(label)


#Harm dimension:

larsen_harm_dir = dimension(semantic_direction_harm)
harm_diagnosis = larsen_harm_dir.cos_sim(diagnosis)
harm_label = larsen_harm_dir.cos_sim(label)


#In-group dimension:

larsen_ingroup_dir = dimension(semantic_direction_ingroup)
ingroup_diagnosis = larsen_ingroup_dir.cos_sim(diagnosis)
ingroup_label = larsen_ingroup_dir.cos_sim(label)


#Fairness dimension:

larsen_fairness_dir = dimension(semantic_direction_fairness)
fairness_diagnosis = larsen_fairness_dir.cos_sim(diagnosis)
fairness_label = larsen_fairness_dir.cos_sim(label)


#Authority dimension:

larsen_authority_dir = dimension(semantic_direction_authority)
authority_diagnosis = larsen_authority_dir.cos_sim(diagnosis)
authority_label = larsen_authority_dir.cos_sim(label)



#Merge all diagnosis dimensions:

diagnosis_cos = [a + b[1:]+ c[1:] + d[1:] + e[1:] for a in purity_diagnosis for b in harm_diagnosis for c in ingroup_diagnosis for d in fairness_diagnosis for e in authority_diagnosis if a[0] == b[0] == c[0] == d[0] == e[0]]
diagnosis = pd.DataFrame(diagnosis_cos, columns = ['word', 'purity', 'harm', 'ingroup', 'fairness', 'authority'])
diagnosis.to_csv('dimensions/diagnosis_cos.8.14.23.csv')


#Merge all label dimensions:

diagnosed_cos = [a + b[1:]+ c[1:] + d[1:] + e[1:] for a in purity_label for b in harm_label for c in ingroup_label for d in fairness_label for e in authority_label if a[0] == b[0] == c[0] == d[0] == e[0]]
label = pd.DataFrame(diagnosed_cos, columns = ['word', 'purity', 'harm', 'ingroup', 'fairness', 'authority'])
label.to_csv('dimensions/labels_cos.8.14.23.csv')


#Read the cosine similarities file for the merged diagnosis and label pairs:
diagnosis_label_cos = pd.read_csv('dimensions/merged_cos.8.14.23.csv')


#Make dodged bar plots from the projected diagnosis-label pairs on the authority moral dimension driven from GPT-3 model:

#Customize font type and size:
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ['Helvetica']
plt.rcParams['font.size'] = 14

# Customize default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#E66100', '#5D3A9B'])

#Make the base plot
fig, ax = plt.subplots(figsize=(19, 8))

# Create a dictionary to store y-axis positions for each group
y_positions = {name: list(range(len(group))) for name, group in diagnosis_label_cos.groupby('type')}

# Group by 'type' (i.e., diagnosis or label) and plot dodged bars
dodge_height = 0.4  # Adjust this value for desired spacing between groups
grouped = diagnosis_label_cos.groupby('type')
n_groups = len(grouped)

# Plot dodged horizontal bars for each group, with labels positioned on the right or left tail based on cosine similarity
for i, (name, group) in enumerate(grouped):
    bars = ax.barh([y + dodge_height * (i - (n_groups - 1) / 2) for y in y_positions[name]], group['Authority'], height=dodge_height, label=name) 
    for bar, word in zip(bars, group['word']):
        xval = bar.get_width()
        #Place bar label at the right tail when cosine similarity is positive
        if xval >= 0:
            ax.text(xval + 0.0001, bar.get_y() + bar.get_height()/2, word, va='center', ha='left')
        #Place bar label at the left tail when cosine similarity is negative
        else:
            ax.text(xval - 0.0001, bar.get_y() + bar.get_height()/2, word, va='center', ha='right')

# Add labels and title
ax.set_xlabel('Authority')
ax.legend()
ax.axvline(x=0, color='black')

# Remove y-axis ticks and labels
ax.yaxis.set_visible(False)

# Remove spines on left, right, and top
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(True)

# Keep only the bottom spine
ax.spines['bottom'].set_visible(True)
ax.xaxis.tick_bottom()

# Add a subtitle at the bottom of the plot
subtitle_text = 'Figure 4: The Projection of Diagnosis-label pairs on Authority Dimension'
ax.text(0.5, -0.15, subtitle_text, transform=ax.transAxes, va='center', ha='center')

# Adjust the bottom margin to prevent cutting off the subtitle
plt.subplots_adjust(bottom=0.2)

# Save the plot as a high-quality pdf file:
plt.savefig('Authority.pdf', format='pdf', bbox_inches='tight')

plt.tight_layout()
plt.show()




