import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
import os

from collections import Counter
import itertools


def calc_chi_square(alpha, label1, label2, study_set1, study_set2):
    cross_table = pd.crosstab(pd.Categorical(study_set1[label1]), pd.Categorical(study_set2[label2]))
    chi2, p_val, dof, _ = stats.chi2_contingency(cross_table)
    message = ''
    if p_val <= alpha:
        message = label1 +' - ' +label2+': Dependent'
    else:
        message = label1 +' - ' +label2+': Independent'
    return (message, chi2, p_val, dof, cross_table)





def calc_correlations_subtopics(labels1, labels2, ignore_diagonal=True, calc_above_diagonal=True):

    table_all = []
    p_val_all = np.zeros((len(labels1), len(labels2)))

    for i in range(len(labels1)):
        table_all.append([])
        for j in range(len(labels2)):
            if ignore_diagonal and j == i: continue
            elif not calc_above_diagonal and j>i: continue

            message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1=labels1[i],
                                                                     study_set1=others_set,
                                                                     label2=labels2[j],
                                                                     study_set2=others_set)
            # table_all[i].append((p_val<alpha,message, chi2, p_val, dof, cross_table))
            #print(cross_table)
            if p_val < alpha:
                table_all[i].append('sig')
            elif p_val < alpha * 2:
                table_all[i].append(p_val)
            else:
                table_all[i].append('None')
            p_val_all[i, j] = p_val
            # print( message, p_val)
            # print(cross_table.to_string())

    np.set_printoptions(precision=2, linewidth=2000, suppress=True)
    # print(p_val_all)
    # print((p_val_all < alpha).astype(np.int))

    table_all2 = table_all.copy()
    for i in range(len(table_all2)):
        table_all2[i] = [labels1[i]] + table_all2[i]
    table_all2 = [[' '] + labels2] + table_all2
    table_all_pandas = pd.DataFrame(table_all2)

    table_cs = []
    for i in range(len(labels1)):
        table_cs.append([])
        for j in range(len(labels2)):
            if ignore_diagonal and j == i: continue
            elif not calc_above_diagonal and j>i: continue

            message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1=labels1[i],
                                                                     study_set1=computer_scientists_study_set,
                                                                     label2=labels2[j],
                                                                     study_set2=computer_scientists_study_set)
            # table_cs[i].append((p_val<alpha,message, chi2, p_val, dof, cross_table))
            if p_val < alpha:
                table_cs[i].append('sig')
            elif p_val < alpha * 2:
                table_cs[i].append(p_val)
            else:
                table_cs[i].append('None')
            p_val_all[i, j] = p_val
            # print('only cs: ', message, p_val)
            # print(cross_table.to_string())

    table_cs2 = table_cs.copy()
    for i in range(len(table_cs2)):
        table_cs2[i] = [labels1[i]] + table_cs2[i]
    table_cs2 = [[' '] + labels2] + table_cs2
    table_cs_pandas = pd.DataFrame(table_cs2)

    table_both_significant = [[] for i in range(len(table_cs))]
    for i in range(len(table_cs)):
        #print(len(table_cs), len(table_cs[i]), len(table_all), len(table_all[i]))
        for j in range(len(table_cs[i])):

            if table_cs[i][j] == 'sig' and table_cs[i][j] == table_all[i][j]:
                table_both_significant[i].append('both')
            elif table_cs[i][j] == 'sig':
                table_both_significant[i].append('cs')
            elif table_all[i][j] == 'sig':
                table_both_significant[i].append('all')
            else:
                table_both_significant[i].append('-')

    for i in range(len(table_both_significant)):
        table_both_significant[i] = [labels1[i]] + table_both_significant[i]
    table_both_significant = [[' '] + labels2] + table_both_significant
    table_both_significant_pandas = pd.DataFrame(table_both_significant)

    return table_all_pandas, table_cs_pandas, table_both_significant_pandas


def calc_frequency(study_set, topics=[], absolute=False):
    frequency_sum = 0
    frequency_list =[]
    for topic in topics:
        frequency = 0
        for subtopic in topic:
            frequency += Counter(study_set[subtopic])['Yes']
        tup = tuple((topic[0][0:3], frequency))
        frequency_list.append(tup)
        frequency_sum += frequency

    if absolute:
        frequency_list.sort(key=lambda tuple: tuple[1], reverse=True)
        return frequency_list, frequency_sum

    topics_frequency_relative = []
    for tup in frequency_list:
        new_tup = (tup[0], round(tup[1]/frequency_sum,3)*100)
        topics_frequency_relative.append(new_tup)

    topics_frequency_relative.sort(key=lambda tuple: tuple[1], reverse=True)
    return topics_frequency_relative



def calc_solved_frequency(study_set, solved_subtopics, absolute=False):
    yes_list =[]
    no_list = []
    yeses= 0
    nos =0
    for topic in solved_subtopics:
        yeses_top = 0
        nos_top = 0
        for subtopic in topic:
            yeses_sub = Counter(study_set[subtopic])['Yes/Kind of']
            nos_sub = Counter(study_set[subtopic])['No']
            yeses+=yeses_sub
            nos += nos_sub
            yeses_top+=yeses_sub
            nos_top+=nos_sub
        yes_tup = tuple((topic[0][0:3], yeses_top))
        no_tup = tuple((topic[0][0:3], nos_top))
        yes_list.append(yes_tup)
        no_list.append(no_tup)

    if absolute:
        yes_list.sort(key=lambda tuple: tuple[1], reverse=True)
        no_list.sort(key=lambda tuple: tuple[1], reverse=True)
        return yes_list, no_list

    yes_list_relative = []
    for a,f in yes_list:
        yes_list_relative.append(tuple((a,round(f/(yeses+nos), 3)*100)))

    no_list_relative = []
    for a,f in no_list:
        no_list_relative.append(tuple((a,round(f/(yeses+nos), 3)*100)))

    no_list_relative.sort(key=lambda tuple: tuple[1], reverse=True)
    return yes_list_relative, no_list_relative


if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    cs_data_set = os.path.join(dir, 'dataset_cs.csv')
    others_data_set = os.path.join(dir, 'dataset_others.csv')

    computer_scientists_study_set = pd.read_csv(cs_data_set, delimiter=',', header=0, encoding='utf8', comment='#', index_col=0)
    others_set = pd.read_csv(others_data_set, delimiter=',', header=0, encoding='utf8', comment='#', index_col=0)

    alpha = 0.05
    print('------------------------------------------------------')
    print('Significance Level :', alpha, ' => [if p_val < ', alpha, ' -> Reject H_0 -> Dependence]')


    # Correlation birth-country & cultural-problems
    message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1='cultural_any_problem', study_set1=others_set, label2='birth_country', study_set2=others_set)
    print('all: ', message, round(p_val,3))
    #print(cross_table.to_string())

    message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1='cultural_any_problem', study_set1=computer_scientists_study_set, label2='birth_country', study_set2=computer_scientists_study_set)
    print('only cs: ', message, round(p_val,3))
    #print(cross_table.to_string())

    # Correlation birth-country germany & cultural problems
    message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1='cultural_any_problem', study_set1=others_set, label2='birth_country_germany', study_set2=others_set)
    print('all: ', message, round(p_val,3))
    #print(cross_table.to_string())

    message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1='cultural_any_problem', study_set1=computer_scientists_study_set, label2='birth_country_germany', study_set2=computer_scientists_study_set)
    print('only cs: ', message, round(p_val,3))
    #print(cross_table.to_string())

    # Correlation birth-country different from study/ work country & cultural problems
    message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1='cultural_any_problem', study_set1=others_set, label2='birth_country_differs_from_work_or_study', study_set2=others_set)
    print('all: ', message, round(p_val,3))
    #print(cross_table.to_string())

    message, chi2, p_val, dof, cross_table = calc_chi_square(alpha=alpha, label1='cultural_any_problem', study_set1=computer_scientists_study_set, label2='birth_country_differs_from_work_or_study', study_set2=computer_scientists_study_set)
    print('only cs: ', message, round(p_val,3))
    #print(cross_table.to_string())

    # Correlation between different subtopics

    subtopics_without_others = ["cultural_religion_problem", "cultural_clothing_problem",
                                "cultural_namechanging_problem", "cultural_infrastructure_problem",
                                "personal_hormones_problem", "personal_lifeplan_problem",
                                "personal_responsibilities_problem", "personal_pregnancy_problem",
                                "work_equality_problem", "work_unsupported_problem", "work_collaborations_problem",
                                "work_infrastructure_problem", "technical_ideasharing_problem",
                                "technical_appreciation_problem", "technical_publications_problem",
                                "technical_researcher_problem"]

    subtopics_all, subtopics_cs, subtopics_both = calc_correlations_subtopics(subtopics_without_others, subtopics_without_others, calc_above_diagonal=False)
    #print(subtopics_all.to_string())
    #print(subtopics_cs.to_string())
    print(subtopics_both.to_string())
    #subtopics_both.to_csv(dir+"\subtopics_correlation_both.csv")

    # Correlation between different topics
    topics_any_probs_labels = ['cultural_any_problem', 'personal_any_problem', 'work_any_problem',
                               'technical_any_problem']

    topics_all, topics_cs, topics_both = calc_correlations_subtopics(topics_any_probs_labels, topics_any_probs_labels, calc_above_diagonal=False)
    #print(topics_all.to_string())
    #print(topics_cs.to_string())
    print(topics_both.to_string())
    #topics_both.to_csv(dir+"\\topics_correlation_both.csv")

    # Correlation between age & subtopics
    subtopics_all, subtopics_cs, subtopics_both = calc_correlations_subtopics(subtopics_without_others, ['age', 'age_regroup'], ignore_diagonal=False)
    #print(subtopics_all.to_string())
    #print(subtopics_cs.to_string())
    print(subtopics_both.to_string())
    #subtopics_both.to_csv(dir+"\subtopics_age_correlation_both.csv")

    #Correlation between age & topics
    topics_all, topics_cs, topics_both = calc_correlations_subtopics(topics_any_probs_labels, ['age', 'age_regroup'], ignore_diagonal=False)
    #print(topics_all.to_string())
    #print(topics_cs.to_string())
    print(topics_both.to_string())
    #topics_both.to_csv(dir+"\\topics_age_correlation_both.csv")

    cultural_prob_labels = ["cultural_religion_problem", "cultural_clothing_problem", "cultural_namechanging_problem", "cultural_infrastructure_problem",
                "cultural_others_problem"]

    personal_prob_labels = ["personal_hormones_problem", "personal_lifeplan_problem", "personal_responsibilities_problem", "personal_pregnancy_problem",
                "personal_others_problem"]

    work_prob_labels = ["work_equality_problem", "work_unsupported_problem", "work_collaborations_problem", "work_infrastructure_problem", "work_others_problem"]

    technical_prob_labels = ["technical_ideasharing_problem", "technical_appreciation_problem", "technical_publications_problem", "technical_researcher_problem",
                 "technical_others_problem"]

    topics = [cultural_prob_labels, personal_prob_labels, work_prob_labels, technical_prob_labels]
    subtopics = cultural_prob_labels + personal_prob_labels + work_prob_labels + technical_prob_labels

    # Most often topics of problems
    all_topics_frequency_relative = calc_frequency(others_set, topics= topics, absolute=False)
    print('All-Topics: ',all_topics_frequency_relative)

    cs_topics_frequency_relative = calc_frequency(computer_scientists_study_set, topics= topics, absolute=False)
    print('CS-Topics: ', cs_topics_frequency_relative)


    cultural_solved_labels = ["cultural_religion_solved", "cultural_clothing_solved", "cultural_namechanging_solved", "cultural_infrastructure_solved",
                "cultural_others_solved"]

    personal_solved_labels = ["personal_hormones_solved", "personal_lifeplan_solved", "personal_responsibilities_solved", "personal_pregnancy_solved",
                "personal_others_solved"]

    work_solved_labels = ["work_equality_solved", "work_unsupported_solved", "work_collaborations_solved", "work_infrastructure_solved", "work_others_solved"]

    technical_solved_labels = ["technical_ideasharing_solved", "technical_appreciation_solved", "technical_publications_solved", "technical_researcher_solved",
                 "technical_others_solved"]

    solved_subtopics_ = cultural_solved_labels + personal_solved_labels + work_solved_labels + technical_solved_labels
    solved_subtopics = [cultural_solved_labels, personal_solved_labels, work_solved_labels, technical_solved_labels]
    # Most often solved topics of problems

    all_solved_frequency, all_not_solved_frequency = calc_solved_frequency(others_set, solved_subtopics)
    summe = sum(f for t,f in all_solved_frequency)
    no_summe = sum(f for t, f in all_not_solved_frequency)
    print('All-Topics-Solved: ', summe, no_summe,all_solved_frequency)

    cs_solved_frequency, cs_not_solved_frequency  = calc_solved_frequency(computer_scientists_study_set, solved_subtopics)
    summe = sum(f for t, f in cs_solved_frequency)
    no_summe = sum(f for t, f in cs_not_solved_frequency)
    print('CS-Topics-Solved: ', summe, no_summe, cs_solved_frequency)



    # Most often subtopics
    subtopics_frequency = [(label, Counter(others_set[label])['Yes']) for label in subtopics]
    sum_frequency_problems_all = sum(f for label,f in subtopics_frequency)
    #print('CS-All-subtopics (absolute): ', subtopics_frequency)
    subtopics_frequency_relative = [(label, round(frequency/sum_frequency_problems_all,3)*100) for label,frequency in subtopics_frequency]
    subtopics_frequency_relative.sort(key= lambda tuple: tuple[1], reverse=True)
    print('All-subtopics: from: ', sum_frequency_problems_all, subtopics_frequency_relative)

    subtopics_frequency = [(label, Counter(computer_scientists_study_set[label])['Yes']) for label in subtopics]
    sum_frequency_problems_cs = sum(f for label,f in subtopics_frequency)
    #print('CS-Subtopics (absolute): ', subtopics_frequency)
    subtopics_frequency_relative = [(label, round(frequency/sum_frequency_problems_cs,3)*100) for label,frequency in subtopics_frequency]
    subtopics_frequency_relative.sort(key= lambda tuple: tuple[1], reverse=True)
    print('CS-Subtopics: from: ',sum_frequency_problems_cs, subtopics_frequency_relative)

    # Most often solved subtopics
    subtopics_frequency = [(label, Counter(others_set[label])['Yes/Kind of']) for label in solved_subtopics_]
    sum_frequency = sum(f for label, f in subtopics_frequency)
    print('CS-All-subtopics-solved (absolute): ', sum_frequency)
    subtopics_frequency_relative = [(label, round(frequency / sum_frequency_problems_all, 3) * 100) for label, frequency in
                                    subtopics_frequency]
    subtopics_frequency_relative.sort(key=lambda tuple: tuple[1], reverse=True)
    print('All-subtopics-solved: from: ', sum_frequency_problems_all, subtopics_frequency_relative)

    subtopics_frequency = [(label, Counter(computer_scientists_study_set[label])['Yes/Kind of']) for label in solved_subtopics_]
    sum_frequency = sum(f for label, f in subtopics_frequency)
    #print('CS-Subtopics-solved(absolute): ', subtopics_frequency)
    subtopics_frequency_relative = [(label, round(frequency / sum_frequency_problems_cs, 3) * 100) for label, frequency in
                                    subtopics_frequency]
    subtopics_frequency_relative.sort(key=lambda tuple: tuple[1], reverse=True)
    print('CS-Subtopics-solved: from: ', sum_frequency_problems_cs, subtopics_frequency_relative)


