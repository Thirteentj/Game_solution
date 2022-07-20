import math
import numpy as np
import pandas as pd
import datetime

'''
内容：对仙霞剑河交叉口的原始数据进行格式处理，合并，以保持和waymo场景中数据格式的一致性
'''


def format_process():
    header_index_list = df_left[df_left[1]=='GlobalTime'].index.tolist()
    for i in range(len(header_index_list)):
        header_index = header_index_list[i]
        if type(df_left[0].iloc[header_index + 1]) == int:
            veh_id = df_left[0].iloc[header_index + 1]
        elif type(df_left[0].iloc[header_index]) == int:
            veh_id = df_left[0].iloc[header_index]
        print(i, veh_id, type(veh_id) == int, f'行数{header_index + 1}')
        if type(veh_id) == int:
            if i < len(header_index_list) - 1:
                next_header_index = header_index_list[i + 1]
                # print(f'next_header_index:{next_header_index}')
                for k in range(header_index, next_header_index):
                    if type(df_left[1].iloc[k]) == datetime.time:
                        df_left[0].iloc[k] = veh_id

            elif i == len(header_index_list) - 1:
                for k in range(header_index, len(df_left)):
                    if type(df_left[1].iloc[k]) == datetime.time:
                        df_left[0].iloc[k] = veh_id

    df_2 = df_left[df_left['GlobalTime'] != 'GlobalTime'] #删除中间的表头

    df_stra.rename(columns={'Unnamed: 0':'ID'},inplace=True)

    a = df_inter['左转ID'].tolist()
    left_id_list = [i  for i in a  if type(i)==int]
    for i in range(len(left_id_list)):
        left_id = left_id_list[i]
        id_index_now = df_inter[df_inter['左转ID']==left_id].index.tolist()[0]
        print(id_index_now)
        if i < len(left_id_list)-1:
            next_left_id = left_id_list[i+1]
            id_index_next = df_inter[df_inter['左转ID']==next_left_id].index.tolist()[0]
            for k in range(id_index_now,id_index_next):
                df_inter['左转ID'].iloc[k] = left_id
        elif i == len(left_id_list) -1 :
            for k in range(id_index_now,len(df_inter)):
                df_inter['左转ID'].iloc[k] = left_id

def info_extract_inter():  #提取交互车的信息：车辆对
    info_list = []
    for id_ in left_id_list:
        dic = {}
        dic['左转ID'] = id_
        # 前车信息记录
        a = df_inter[(df_inter['左转ID'] == id_) & (df_inter['穿越前车'] == 1)]
        if len(a) > 0:
            dic['穿越前车'] = int(a['直行ID'])
            if len(a['PET（s）'].tolist()) > 0:
                pet_a = a['PET（s）'].tolist()[0]
                if (type(pet_a) == float) and (np.isnan(pet_a) == False):
                    dic['前车PET'] = pet_a
            if len(a['TTC（s）'].tolist()) > 0:
                ttc_a = a['TTC（s）'].tolist()[0]
                if (type(ttc_a) == float) and (np.isnan(pet_a) == False):
                    dic['前车TTC'] = ttc_a

        # 后车信息记录
        b = df_inter[(df_inter['左转ID'] == id_) & (df_inter['穿越后车'] == 1)]
        if len(b) > 0:
            dic['穿越后车'] = int(b['直行ID'])
            if len(b['PET（s）'].tolist()) > 0:
                pet_b = b['PET（s）'].tolist()[0]
                if (type(pet_b) == float) and (np.isnan(pet_b) == False):
                    dic['后车PET'] = pet_b
            if len(b['TTC（s）'].tolist()) > 0:
                ttc_b = b['TTC（s）'].tolist()[0]
                if (type(ttc_b) == float) and (np.isnan(pet_b) == False):
                    dic['后车TTC'] = ttc_b

            # if (dic['穿越前车'] in left_id_list or dic['穿越前车'] in stra_id_list) and \
            #         (dic['穿越后车'] in left_id_list or dic['穿越后车'] in stra_id_list):
            info_list.append(dic)

    df_info = pd.DataFrame(info_list)
    return df_info
def info_extract_inter_V2():  #提取交互车的信息：车辆对,简化版信息提取
    info_list = []
    for id_ in left_id_list:
        dic = {}
        dic['左转ID'] = id_
        # 后车信息记录
        b = df_inter[(df_inter['左转ID'] == id_) & (df_inter['穿越后车'] == 1)]
        if len(b) > 0:
            dic['穿越后车'] = int(b['直行ID'])
            if len(b['PET（s）'].tolist()) > 0:
                pet_b = b['PET（s）'].tolist()[0]
                if (type(pet_b) == float) and (np.isnan(pet_b) == False):
                    dic['后车PET'] = pet_b
            if len(b['TTC（s）'].tolist()) > 0:
                ttc_b = b['TTC（s）'].tolist()[0]
                if (type(ttc_b) == float) and (np.isnan(pet_b) == False):
                    dic['后车TTC'] = ttc_b

            if (dic['左转ID'] in left_id_list_ori and dic['穿越后车'] in stra_id_list_ori):
                info_list.append(dic)
    df_info = pd.DataFrame(info_list)
    return df_info

def time_shift(time_str):
    if np.isnan(time_str):
        return -1
    else:
        if len(time_str) > 12:
            a = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
        else:
            a = datetime.datetime.strptime(time_str, "%H:%M:%S")
        return a




# filepth1 = r'D:\BaiduSyncdisk\仙霞剑河数据\南进口左转机动车筛选数据.xlsx'  #左转直行车的交互信息 交叉口西进口左转车的轨迹数据
# # filepth2 = r'D:\BaiduSyncdisk\仙霞剑河数据\南进口左转机动车筛选数据.xlsx'  #
# filepth3 = r'D:\BaiduSyncdisk\仙霞剑河数据\北进口直行机动车.xlsx'  #直行车的轨迹数据
#
# df_inter = pd.read_excel(filepth1,sheet_name='Sheet3')  #交互信息记录
# df_left = pd.read_excel(filepth1,sheet_name='Sheet1')
# df_stra = pd.read_excel(filepth3)
# #左车轨迹ID  从原始轨迹集中得到
# left_id_list_ori = pd.unique(df_left['ID'])
# stra_id_list_ori = pd.unique(df_stra['ID'])
# #从交互信息表中提取
# a = df_inter['左转ID'].tolist()
# left_id_list = list(set([i  for i in a]))
# left_id_list.sort()
# df_info = info_extract_inter()


#数据内部处理：waymo数据对齐，时间帧对齐，交互范围对齐

def frame_label_process():
    df_all_veh = pd.DataFrame()
    for i in range(len(df_info)):

        left_id = df_info.iloc[i]['左转ID']
        stra_id = df_info.iloc[i]['穿越后车']
        print(f'这是第{i}个片段,左转片段ID{left_id}，直行片段ID{stra_id}')
        df_left_single = df_left[df_left['ID']==left_id]
        df_stra_single = df_stra[df_stra['ID']==stra_id]
        df_left_single = df_left_single.reset_index(drop=True)
        df_stra_single = df_stra_single.reset_index(drop=True)
        df_left_single.insert(0,'segment_index',0)
        df_stra_single.insert(0, 'segment_index',0)
        df_left_single.insert(1,'scenario_label',i)
        df_stra_single.insert(1, 'scenario_label',i)
        df_left_single.insert(2,'frame_label',-1)
        df_stra_single.insert(2, 'frame_label',-1)
        df_left_single.insert(3, 'time_stamp', -1)
        df_stra_single.insert(3, 'time_stamp', -1)
        df_left_single.insert(4, 'action_type', 'left')
        df_stra_single.insert(4, 'action_type', 'straight')
        df_left_single.insert(5, 'length', 5.5)
        df_stra_single.insert(5, 'length', 5.5)
        df_left_single.insert(6, 'width', 2.3)
        df_stra_single.insert(6, 'width', 2.3)
        left_time_min,stra_time_min = np.min(df_left_single['GlobalTime']),np.min(df_stra_single['GlobalTime'])
        left_time_min = time_shift(left_time_min)  #转换为datetime 格式
        stra_time_min = time_shift(stra_time_min)
        if left_time_min < stra_time_min:  #先对左转车序列赋值
            df_left_single['frame_label'] = df_left_single.index
            #寻找与另一辆车最接近的时间戳
            left_time_list = df_left_single['GlobalTime']
            min_time_diff = abs(stra_time_min-left_time_min)
            min_time_index = -1
            for k in range(len(left_time_list)):
                a = time_shift(left_time_list[k])
                time_diff = abs(stra_time_min-a)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    min_time_index = k
            df_stra_single['frame_label'] = df_stra_single.index + min_time_index

        elif left_time_min>stra_time_min:  #先对直行车序列赋值
            df_stra_single['frame_label'] = df_stra_single.index
            # 寻找与另一辆车最接近的时间戳
            stra_time_list = df_stra_single['GlobalTime']
            min_time_diff = abs(stra_time_min - left_time_min)
            min_time_index = -1
            for k in range(len(stra_time_list)):
                a = time_shift(stra_time_list[k])
                time_diff = abs(left_time_min - a)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    min_time_index = k
            df_left_single['frame_label'] = df_left_single.index + min_time_index
        else:
            df_left_single['frame_label'] = df_left_single.index
            df_stra_single['frame_label'] = df_stra_single.index
        df_left_single['time_stamp'] = df_left_single['frame_label'] * 0.1
        df_stra_single['time_stamp'] = df_stra_single['frame_label'] * 0.1
        df_all_veh = pd.concat([df_all_veh,df_left_single,df_stra_single])

    df_all_veh.insert(7,'heading',-1)
    df_all_veh['heading'] = np.arctan((df_all_veh['Vy[m/s]']/df_all_veh['Vx[m/s]']))

    df_all_veh.rename(columns={'ID': 'obj_id', 'X[m]': 'center_x', 'Y[m]': 'center_y',
                               'Vx[m/s]': 'velocity_x', 'Vy[m/s]': 'velocity_y',
                               'Ax[m/s2]':'ax_next','Ay[m/s2]':'ay_next'
                               }, inplace=True)
    outpath_4 = r'C:\Users\刘佳琦\Desktop\毕设内容\仙霞剑河数据\veh_all_info_tra.xlsx'
    df_all_veh.to_excel(outpath_4, index=None)


filepth1 = r'D:\BaiduSyncdisk\waymo-od\veh_all_info_tra.xlsx'  #左转直行车的交互信息 交叉口西进口左转车的轨迹数据
filepth2 = r'D:\BaiduSyncdisk\waymo-od\veh_all_info_tra_2.xlsx'  #
outpath = r'D:\BaiduSyncdisk\waymo-od\veh_all_info_tra_all.xlsx'
#合并Excel数据
df_1 = pd.read_excel(filepth1)
df_2 = pd.read_excel(filepth2)
df_1.insert(1,'direction','west')
df_2.insert(1,'direction','south')
df_all = pd.concat([df_1,df_2])
df_all.to_excel(outpath,index=None)











