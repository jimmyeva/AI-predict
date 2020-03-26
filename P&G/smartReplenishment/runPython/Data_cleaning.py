## this for Data cleaning
import Parameters
import numpy as np
import pandas as pd

class CleanData():
    def __init__(self):
        return

## clean data for missing value
    def missing_value(self,data_frame):
        dataframe0 = data_frame.fillna(0)
        return dataframe0

    ## clean data for combine data(sum/mean/count) and sort
    def combine_data(self,data_frame, GroupIndex, CombineIndex, Fun, SortIndex):
        group_frame = data_frame.groupby(GroupIndex, as_index= False)
        combine_frame = pd.DataFrame(group_frame[CombineIndex].agg(Fun))
        combine_frame1 = pd.DataFrame(combine_frame.sort_values(SortIndex,ascending=True))
        return combine_frame1

    ## add a column named week_day
    ## add a column named week_day, time_frame from dataframe (time column)
    def to_weekday(self, time_frame):
        date0 = pd.to_datetime(time_frame)
        week_day_array0 = np.array(list(map(lambda x: x.weekday(), date0)))
        return week_day_array0

    # delect the values which have NA or N/A
    def del_NA(self, data_frame, col_name):
        NA_index = ['[NULL]','NA', 'N/A']
        data_frame0 = data_frame[~data_frame[col_name].isin(NA_index)]
        return  data_frame0

    # transform the values to str from different kind of characters
    def to_str(self, data_frame, col_name):
        data_frame[col_name] = data_frame[col_name].astype(str)
        return data_frame

    # get the intersection from different set,input:two set,output:list
    def diff(self ,A,B):
        diff = A.intersection(B)
        return list(diff)

    ## add the  assist column help distinct the circulation
    def agg_to_frame(self,data_frame, t3):
        num0 = int(data_frame.shape[0] / t3)
        p_days = []
        for i in range(num0):
            days = np.repeat(i, t3)
            p_days.append(days)
        data_frame['days'] = np.array(p_days).ravel()
        return data_frame

    # use the sigma to amend the Error_sum_days
    def Err_sigma_V1(self,Error_sum_days, sigma):
        Error_sum_days1 = []
        for i in Error_sum_days:
            if i > 0:
                li = i * sigma
            else:
                li = i / sigma
            Error_sum_days1.append(li)
        return Error_sum_days1

    # Safe inventory calculations(prediction for t2 days plus The uncertainty of prediction----error)
    # 1 prediction sum for t2 days
    def Pre_Error_sum(self,Frame ,column_index ,t2, t3):
        Sum_days = []
        column_name = list(Frame.columns)
        Index = column_name.index(column_index)
        num0 = int(Frame.shape[0] / t3)
        for i in range(num0):
            Sum_days.append(sum(Frame[Frame['days']==i].iloc[(t2):(2*t2),Index]))
        return Sum_days