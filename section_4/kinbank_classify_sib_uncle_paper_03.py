# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:53:48 2019
kinbank_classify_sib_uncle_paper_03.py
@author: Wolfgang
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

datafile = './section_4/results/data_dict.txt'
outfile = './section_4/results/clasification.csv'

heatmapImage = "./section_4/results/cousins_uncles.png"
counting_dict = {}

print(os.getcwd())

# male ego
haw_1 = [["mF", "mFeB", "mMeB"]] # father and both uncles are the same
esk_1 = [["mF"], ["mFeB", "mMeB"]]  # all uncles are the same but father is different
dra_1 = [["mF", "mFeB"], ["mMeB"]] # father and father's brother are the same, mother's brother is different
sud_1 =  [["mF"], ["mFeB"], ["mMeB"]] # all male +1 generation are different

haw_0 = [['meB', 'exmFBSx', 'exmMBSx']] # all cousins are like all siblings
esk_0 = [['meB'], ['exmFBSx', 'exmMBSx']] # all cousins are the same but are different to siblings
dra_0 = [['meB', 'exmFBSx'], ['exmMBSx']] # parallel cousins are like siblings, cross cousins are different
sud_0 = [['meB'], ['exmFBSx'], ['exmMBSx']]  # all cousins and siblings are different

heads = ['dataID', 'Language', 'mF', 'mFeB', 'mMeB', 'meB', 'exmFBSx', 'exmMBSx', 'parentsCategory', 'cousinsCategory']

groupsDict = {'haw_1': haw_1, #uncles
              'sud_1': sud_1, 
              'esk_1': esk_1,
              'dra_1': dra_1,
              'haw_0': haw_0, # brothers & cousins
              'sud_0': sud_0, 
              'esk_0': esk_0,
              'dra_0': dra_0,
}

cousins = {'haw_0': haw_0, # brothers & cousins
           'sud_0': sud_0, 
           'esk_0': esk_0,
           'dra_0': dra_0}

uncles = {'haw_1': haw_1, #uncles
          'sud_1': sud_1, 
          'esk_1': esk_1,
          'dra_1': dra_1}

def read_dict(file):
    # read a dictionary
    s = open(file, 'r',  encoding="utf-8").read()
    dataDict = eval(s)
    return dataDict

def fill_in_terms(v, b):
    
    """takes the group list and looks up each of the terms
    returns the list filled
    """
    
    newList = [] 
    
    for group in b:
        
        newList.append([])
        
        for each in group:
                                        
            if v[each][0] not in newList[-1]:
                newList[-1].append(v[each][0])
    return newList
  
    
def filter_data(newList):
    
    """ check for indicators that the data is no match to the pattern
        by default, a kinship system is systemPosotive: True, unless
        it fails one of the criteria
    """
                
    systemPositive = True
    
    for group in newList:

        if len(group) > 1:  # if there is more than one term its not a syncretism
            systemPositive = False

        if '#' in group:    # if there is a hashtag, the datapoint is missing
            systemPositive = False
    
    # check if one term is in more than one subgroup, which indicates that the
    # syncretism is somewhere else.

    if len(newList) > 1:
        
        checkList = []
        for each in newList:
            if each not in checkList:
                checkList.append(each)
                    
        if len(checkList) < len(newList):
            systemPositive = False
        #print (checkList)

    return systemPositive                     
    

def make_heatmap(counting_dict):
    """ take the data dictionary turns it into a pandas matrix and
        prints a colorcoded heatmap of the matrix
    """
    
    #plotTitle = v['Family']
    #plotTitle = v['Macroarea']
    plotTitle = 'Kin Categories'
    
    for k, v in counting_dict.items():
        numbi = 0
        for a, b in v.items():
            numbi += b
        
    parentDict = {}
    for k, v in counting_dict.items():
        
        numbi = 0
        for a, b in v.items():
            if a in parentDict:
                parentDict[a] += b
            else:
                parentDict[a] = b
        
    df = pd.DataFrame(counting_dict)
    
    #df = df.reindex(sorted(df.columns), axis=1)
    
    new_index = ['esk_1', 'haw_1', 'dra_1', 'sud_1']
    df = df.reindex(new_index)
    df.rename(index={'esk_1' : 'Eskimo', 'haw_1':'Hawaiian', 'dra_1': 'Dravidian', 'sud_1': 'Sudanese'}, inplace=True)
    
    new_index_column = ['sud_0', 'dra_0', 'haw_0', 'esk_0']
    df = df.reindex(columns = new_index_column)
    df.columns = ['Sudanese', 'Dravidian', 'Hawaiian', 'Eskimo']
    
    #df = df.div(df.sum(axis=0), axis=1) * 100 # normalise by column
    #df = df.div(df.sum(axis=1), axis=0)* 100 # normalise by row

    df = df.round(1)  
    print (df)
    
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.set(font_scale=1.8)
    
    g = sns.heatmap(df, cmap="Blues", annot=True, fmt='g', ax=ax, cbar=False) # annot_kws={"fontsize":16}
    
    plt.ylabel('GENERATION +1', labelpad=15)
    plt.xlabel('GENERATION 0', labelpad=15)
    
    plt.title(plotTitle, fontsize='30')
    
    g.get_figure().savefig(heatmapImage)
 
    

def populated_empty(counting_dict):
    """ add a key with value 0 for every dictionary entry
    """
    for v, b in cousins.items():
        for k, h in uncles.items():
            if v not in counting_dict:
                counting_dict[v] = {}
            counting_dict[v][k] = 0
    
    return counting_dict
    

def find_relatives(cousins, v):
    for c, d in cousins.items():
        newList = fill_in_terms(v, d)
        systemPositive = filter_data(newList)
        #print (newList)
        if systemPositive:
    
            return c

    return False

def write_to_file(k, v):
    """ write patterns to file
    """
            
    outf.write(k + '\t' + v['displayName'] + '\t')
        
    for x in ['mF', 'mFeB', 'mMeB', 'meB', 'exmFBSx', 'exmMBSx']:
        outf.write(str(v[x]) + '\t')
    
    outf.write(str(v['uncleSystem']) + '\t' + str(v['cousinSystem']) + '\t')
    outf.write('\n')


def count_patterns(counting_dict, count):
    
    """ increase value in counting_dict if both systems are present
    """
    
    if (v['uncleSystem'] != False) and (v['cousinSystem'] != False):

        counting_dict[v['cousinSystem']][v['uncleSystem']] += 1
        count += 1

    return counting_dict, count

if __name__== "__main__":
    
    with open(outfile, "w", encoding="utf-8") as outf:
        outf.write('\t'.join(heads)+ '\n')
    
        countAll = 0
        count = 0
        
        counting_dict = populated_empty(counting_dict)
        
        rawData = read_dict(datafile) # read the dictionary from the file
        
        for k, v in rawData.items():
            # this can be used as a filter
            #if v['Family'] == 'Austronesian':
            #if v['Family'] == 'Algic': 
            #if v['Family'] == 'Atlantic-Congo':
            #if v['Family'] == 'Pama-Nyungan':
            #if v['Family'] == 'Indo-European':
            #if v['Family'] not in ['Algic', 'Atlantic-Congo', 'Pama-Nyungan', 'Austronesian', 'Indo-European']:
            #if v['Macroarea'] == 'Africa':
            #if v['Macroarea'] == 'North America':
            #if v['Macroarea'] == 'South America':
            #if v['Macroarea'] == 'Australia':
            #if v['Macroarea'] == 'Eurasia':
            #if v['Macroarea'] == 'Papunesia':
        
            countAll += 1
    
            v['cousinSystem'] = find_relatives(cousins, v) 
            
            v['uncleSystem'] = find_relatives(uncles, v) 
 
            write_to_file(k, v)

            counting_dict, count = count_patterns(counting_dict, count)
            
    outf.close()

    make_heatmap(counting_dict)
    
    print (str(countAll) + ' languages in raw data')
    print (str(count) + ' languages with +1 and 0 generation data')
