import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import math

#计算两个风险：全域总感知风险和交互车辆车辆视角的总风险
#全域总感知风险（当下时刻+未来时刻）：动态对象+静态风险
#交互车辆视角下的总风险：当下时刻全域的总风险和未来时刻的目标交互车的风险
class veh_now():  #一辆车为一个对象
    def __init__(self,id,loc,v_xy,size,time_index=-1):
        self.id = id
        self.loc = loc  #车辆的xy坐标
        self.vx = v_xy[0]
        self.vy = v_xy[1]
        self.v = np.sqrt(self.vx**2+self.vy**2)
        self.size = size  #车辆的尺寸

class map_para_init():
    def __init__(self,x_start,x_end,y_start,y_end):
        #地图范围
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        #x、y方向的分割粒度
        self.x_step = 2*int(abs((self.x_end-self.x_start)))
        self.y_step = 2*int(abs((self.y_end-self.y_start)))
        self.xx = np.linspace(self.x_start, self.x_end, self.x_step)   #网格数据划分准备
        self.yy = np.linspace(self.y_start, self.y_end, self.y_step)
    def map_size(self):
        #绘图尺寸
        self.width = 20*(self.x_end-self.x_start)
        self.height = 20*(self.y_end-self.y_start)
        return self.height,self.width
    def map_meshgrid(self):

        self.X ,self.Y = np.meshgrid(self.xx,self.yy)  #网格划分
        return self.X,self.Y

def cosVector(x,y):  #求两向量的余弦值

    if y[0] == 0 and y[1] == 0:
        cos_ang = 1
    else:
        result1=0.0
        result2=0.0
        result3=0.0
        for i in range(len(x)):
            result1+=x[i]*y[i]  #sum(X*Y)
            result2+=x[i]**2   #sum(X*X)
            result3+=y[i]**2   #sum(Y*Y)

        cos_ang = result1/(np.sqrt(result2) * np.sqrt(result3))
    # if math.isnan(cos_ang):
    #     print(x, y, result1, result2, result3,cos_ang)
    return cos_ang
#cosVector([2,1],[1,1])

def risk_single_dynamic_obj_cal(veh_i,map_para):  #某个时刻单个车辆的风险计算
    x_obj, y_obj = veh_i.loc
    l_obj,w_obj = veh_i.size
    X, Y = map_para.map_meshgrid()
    # print('x,y')
    # print(X)
    # print(Y)
    β_x = α_x = 0.5
    β_y = α_y = 0.5
    #x方向上的衰减因子
    delta_x = β_x * np.max(X-x_obj+0.5*l_obj,0)/(α_x*veh_i.vx+1)
    delta_y = β_y * np.max(Y-y_obj+0.5*w_obj,0)/(α_y*veh_i.vy+1)
    vector_x1 = (X-x_obj,Y-y_obj)
    vector_x2 = (veh_i.vx,veh_i.vy)
    delta_xy = np.sqrt(delta_x**2 + delta_y **2)* np.exp(cosVector(vector_x1,vector_x2))/np.e #cos 保证场的非对称性
    R_dyna_single = 1.0/(delta_xy+1)
    # print('single')
    # print(R_dyna_single)
    if True in (R_dyna_single<0):
        print('error!!!')
    return R_dyna_single
def risk_dynamic_cal(veh_list,map_para):
    R_dyna_all = np.zeros((map_para.y_step, map_para.x_step))
    for veh_i in veh_list:
        R_dyna_all += risk_single_dynamic_obj_cal(veh_i, map_para)
    return R_dyna_all

def risk_all_field_now_cal(veh_list,map_para):  #计算当下时刻的全域总风险
     #全域总风险
    R_all_now = R_all_dynamic_now = R_all_static_now = np.zeros((map_para.y_step, map_para.x_step))
    #分别计算所有动态对象的风险总和以及动态对象的风险总和
    R_all_dynamic_now = risk_dynamic_cal(veh_list,map_para)
    # print('all_dyna',R_all_dynamic_now)
    R_all_now = R_all_dynamic_now + R_all_static_now  #静态风险和动态风险之和
    return R_all_now

def veh_loc_trans(veh_i,dis_,v_):
    trans_xy = veh_i.tra_real_xy
    trans_s_sum = veh_i.tra_real_s_sum
    index = -1
    for i in range(len(trans_s_sum)-1):
        if (trans_s_sum[0][i] < dis_) and (trans_s_sum[0][i + 1] > dis_):
            index = i
            break
    ang = trans_s_sum[1][i]
    v_x_i_next, v_y_i_next = v_ * np.cos(ang) , v_ * np.sin(ang)
    return trans_xy[index][0],trans_xy[index][1],v_x_i_next,v_y_i_next

def risk_interact_field_feature_cal(time_now,veh_L,veh_S,v_L_next, a_L_next,
                                                     v_S_next, a_S_next,map_para):  #计算未来时刻目标交互车的风险
    R_interact_feature_L = R_interact_feature_S = np.zeros((map_para.y_step, map_para.x_step))
    #计算直行车预测的直行车的风险
    dis_s, v_s_now = veh_S.dis_plan[-1], veh_S.v[-1]
    dts = 0.5 * dis_s / v_s_now
    dts = 0.5  #暂时设置为0.5s
    dis_s_pre = dis_s - v_s_now * dts -0.5 * a_S_next * dts ** 2
    v_next_s = v_s_now + a_S_next * dts
    loc_x_s_pre, loc_y_s_pre,v_x_s_next,v_y_s_next = veh_loc_trans(veh_S, dis_s_pre, v_next_s)  #将自然坐标系信息转化为xy坐标系

    veh_S_pre = veh_now(veh_S.id, (loc_x_s_pre, loc_y_s_pre), (v_x_s_next, v_y_s_next),veh_S.size)
    R_interact_feature_S = risk_single_dynamic_obj_cal(veh_S_pre, map_para)
    #计算左车预测的左车的风险
    dis_L, v_L_now = veh_L.dis_plan[-1], veh_L.v[-1]
    dtL = 0.5 * dis_L / v_L_now
    dtL = 0.5  # 暂时设置为0.5s
    dis_L_pre = dis_L - v_L_now * dtL - 0.5 * a_L_next * dtL ** 2
    v_next_l = v_L_now + a_L_next * dts
    loc_x_l_pre, loc_y_l_pre,v_x_l_next, v_y_l_next = veh_loc_trans(veh_L, dis_L_pre, v_next_l)

    veh_L_pre = veh_now(veh_L.id, (loc_x_l_pre, loc_y_l_pre), (v_x_l_next, v_y_l_next),veh_L.size)
    R_interact_feature_L = risk_single_dynamic_obj_cal(veh_L_pre, map_para)

    return R_interact_feature_L,R_interact_feature_S

def process_data(index,top_view_trj, time_now,veh_L,veh_S):
    # 获取全局车辆信息
    veh_list = []
    seg_trj_single = top_view_trj  # 一个20s场景中的所有轨迹数据
    all_veh_info = seg_trj_single[seg_trj_single['time_stamp'] == time_now]  #当下时刻的所有车辆信息
    id_veh_left, id_veh_strai = veh_L.id, veh_S.id
    size_veh_left = veh_L.size
    size_veh_stra = veh_S.size
    for i in range(len(all_veh_info)):
        veh_info = all_veh_info.iloc[i]
        v_id = veh_info.obj_id
        if (v_id != id_veh_left ) and (v_id != id_veh_strai):  #初始化处交互车辆之外的所有车辆信息
            loc_x = veh_info.center_x
            loc_y = veh_info.center_y
            v_x = veh_info.velocity_x
            v_y = veh_info.velocity_y
            len_veh = veh_info.length
            width_veh = veh_info.width
            veh_i = veh_now(v_id, (loc_x, loc_y), (v_x, v_y),(len_veh,width_veh),time_now)
            veh_list.append(veh_i)

    L_x_plan_now,L_y_plan_now,v_x_l_now, v_y_l_now = veh_L.tra_real_xy[0][index],veh_L.tra_real_xy[1][index],\
                                                    veh_L.vx_real[index],veh_L.vy_real[index]

    S_x_plan_now, S_y_plan_now, v_x_s_now, v_y_s_now = veh_S.tra_real_xy[0][index],veh_S.tra_real_xy[1][index],\
                                                    veh_S.vx_real[index],veh_S.vy_real[index]
    veh_left = veh_now(veh_L.id,(L_x_plan_now,L_y_plan_now),
                       (v_x_l_now, v_y_l_now),size_veh_left,time_now)
    veh_strai = veh_now(veh_S.id,(S_x_plan_now, S_y_plan_now),
                       (v_x_s_now, v_y_s_now),size_veh_stra,time_now)
    veh_list.append(veh_left)
    veh_list.append(veh_strai)
    # print('共{}辆车辆'.format(len(veh_list)))
    # 获取全局地图信息
    map_x_min, map_x_max = seg_trj_single['center_x'].min(), seg_trj_single['center_x'].max()
    map_y_min, map_y_max = seg_trj_single['center_y'].min(), seg_trj_single['center_y'].max()
    map_para = map_para_init(map_x_min, map_x_max, map_y_min, map_y_max)
    return veh_list, map_para

def get_risk_interactive_all(index,time_now,veh_L,veh_S,v_L_next, a_L_next,
                                                     v_S_next, a_S_next,seg_trj_single):

    veh_list,map_para = process_data(index,seg_trj_single, time_now, veh_L, veh_S)

    #当下时刻的所有车辆数据
    #  定义权重系数
    a1 = 0.5
    # Risk_L, Risk_S = np.zeros((map_para.y_step, map_para.x_step))
    R_all_now = risk_all_field_now_cal(veh_list,map_para)  #当下时刻的总风险
    # print('all')
    # print(R_all_now)
    R_interact_feature_L,R_interact_feature_S = risk_interact_field_feature_cal(time_now,veh_L,veh_S,v_L_next, a_L_next,
                                                     v_S_next, a_S_next,map_para)  #计算未来时刻目标交互车的风险
    # if np.max(R_all_now) > 1:
    #     print('max is more than 1')
    #     print('time now {}'.format(time_now))
    #     print(R_all_now)

    Risk_L = a1 * R_all_now + (1-a1) * R_interact_feature_L
    Risk_S = a1 * R_all_now + (1-a1) * R_interact_feature_S

    return Risk_L,Risk_S,map_para

