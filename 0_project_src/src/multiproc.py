import os
import pandas as pd
from multiprocessing import Process
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix

def get_matrix(df):
    df = df[['user', 'hotel id', 'ratings']]
    pdf = pd.pivot_table(df,index=['user'], columns = 'hotel id', values = "ratings").fillna(0)
    mat = csr_matrix(pdf)
    return mat

def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    #proc_name = os.getid()
    return col_normed_mat.T * col_normed_mat

if __name__ == '__main__':
    reg = pd.read_csv('dataset/major_region_ratings.csv', index_col=False)
    user_split = reg.groupby('user country')
    sub_user_geo = [user_split.get_group(x) for x in user_split.groups]
    procs = []
    # [['user', 'hotel id', 'ratings']]
    for index, country in enumerate(sub_user_geo):
        proc = Process(target=get_matrix, (sub_user_geo,))
        procs.append(proc)
        proc.start()
        print proc

    for proc in procs:
        proc.join()
