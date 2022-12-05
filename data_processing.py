# Processing data
from pandas import *
import numpy as np
import os

def process_excel_file(path, sheet):
    read = read_excel(path, sheet, index_col=None, na_values=['NA'])
    df = DataFrame(read)[:3]
    # breakpoint()
    df = df[df.columns[:3]]
    return df


def write_into_text(writePath, label, df):
    with open(writePath, 'w') as f:
        for line in df[df.columns[0]]:
            f.write(line + '\t' + str(label) + '\n')
            f.flush()
        f.close()

def combine_common_files(file1,file2, final):
    r1 = open(file1, 'r')
    r2 = open(file2, 'r')
    lines1 = r1.readlines()
    lines2 = r2.readlines()
    questions_list = []
    for l in lines1:
        l = l.strip()
        if l not in questions_list:
            questions_list.append(l)
    for l in lines2:
        l = l.strip()
        if l not in questions_list:
            questions_list.append(l)
    w = open(final, 'w')
    for q in questions_list:
        w.write(q + '\n')
        w.flush()
    w.close()

def write_data(main_path, writePath, addtional_data, sheet, label, file2, final):
    # get the data of chatbot
    df = process_excel_file(main_path + addtional_data, sheet)
    write_into_text(main_path + writePath, label, df)

    # combine chatbot data
    file1 = main_path + writePath
    file2 = main_path + file2
    final = main_path + final
    combine_common_files(file1, file2, final)

def delete_data(main_path, writePath):
    # delete data
    os.remove(main_path + writePath)