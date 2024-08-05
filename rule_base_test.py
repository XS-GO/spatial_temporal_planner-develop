""" 
Filename    : spatial_temporal_planner-develop.py
Author      : shun.xing
Email       : 2473372515@qq.com
Date        : 2024-08-05
Description : Design rules, calculate target points, obtain multiple decision target points, and then use DWA algorithm to sample in time and space to obtain rough solutions
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import copy
import math
from enum import Enum
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle

class Config:
    def __init__(self):
        self.ego_s                = 0
        self.ego_l                = 2
        self.ego_t                = 0 
        self.ego_v_l              = 0
        self.ego_v_s              = 7       
        self.ego_width            = 1.5
        self.ego_length           = 2.5
        self.obs1_s               = 10
        self.obs1_l               = 6 
        self.obs1_t               = 0 
        self.obs1_v_l             = -0.5
        self.obs1_v_s             = 4
        self.obs1_width           = 1.5
        self.obs1_length          = 2.5
        self.obs2_s               = 30
        self.obs2_l               = 2 
        self.obs2_t               = 0 
        self.obs2_v_l             = 0
        self.obs2_v_s             = 6
        self.obs2_width           = 1.5
        self.obs2_length          = 2.5
        self.obs1_heading         = 0
        self.obs2_heading         = 0
        self.delta_t              = 0.1
        self.t_total              = 10
        self.acc_max              = 3
        self.acc_comfort          = 1
        self.dcc_max              = -3
        self.dcc_comfort          = -1
        self.l_safe               = 1


class Update:
    def __init__(self):
        self.config = Config()
        self.reset()
        self.current_t = 0
        self.x = np.array([0.0, 2.0, 0.0, 7, 0.0])  # 初始化自车位置
        self.u = np.array([7, 0.0])  # 初始化控制输入
        self.trajectory = np.array([self.x])  # 初始化轨迹

    def reset(self):
        self.obs1_s = [self.config.obs1_s]
        self.obs1_l = [self.config.obs1_l]
        self.obs1_t = [self.config.obs1_t]
        self.obs2_s = [self.config.obs2_s]
        self.obs2_l = [self.config.obs2_l]
        self.obs2_t = [self.config.obs2_t]

    def update(self):
        self.current_t += self.config.delta_t
        self.obs1_s.append(self.obs1_s[-1] + self.config.obs1_v_s * self.config.delta_t)
        self.obs1_l.append(self.obs1_l[-1] + self.config.obs1_v_l * self.config.delta_t)
        self.obs1_t.append(self.obs1_t[-1] + self.config.delta_t)
        
        self.obs2_s.append(self.obs2_s[-1] + self.config.obs2_v_s * self.config.delta_t)
        self.obs2_l.append(self.obs2_l[-1] + self.config.obs2_v_l * self.config.delta_t)
        self.obs2_t.append(self.obs2_t[-1] + self.config.delta_t)

    def get_obstacle_positions(self):
        obstacles = [
            [self.obs1_s[-1], self.obs1_l[-1],self.current_t],
            [self.obs2_s[-1], self.obs2_l[-1],self.current_t]
        ]
        return np.array(obstacles)


class Rulebase:
    def __init__(self):
        self.obs1_class      = ['non_obs']
        self.obs2_class      = ['line_obs']
        self.obs1            = [config_.obs1_s,config_.obs1_l,config_.obs1_t,config_.obs1_v_s,config_.obs1_v_l]
        self.obs2            = [config_.obs2_s,config_.obs2_l,config_.obs2_t,config_.obs2_v_s,config_.obs2_v_l]
        self.obs_list        = [self.obs1,self.obs2]
    
    def calculate_ttc(self, obs_s, obs_v_s,obs_l,obs_v_l):

        ateral_distance = abs(config_.ego_l - obs_l)
        l_ttc = float('inf')
        if config_.ego_v_s > obs_v_s and obs_v_l <= 0:
            s_ttc = -(config_.ego_s - obs_s) / (config_.ego_v_s - obs_v_s)
            if config_.ego_v_l - obs_v_l !=0:
                l_ttc = ateral_distance / (config_.ego_v_l - obs_v_l)
            return min(s_ttc,l_ttc)
        else:
            return float('inf')


    def select_target_obs(self):
        min_ttc = float('inf') 
        target_obs = None
        for obs_tuple in self.obs_list:
            
            obs_s, obs_l, obs_t,obs_v_s, obs_v_l = obs_tuple
            if(obs_l!=config_.ego_l):
                if(obs_v_l>=0):
                    continue
            ttc = self.calculate_ttc(obs_s, obs_v_s,obs_l,obs_v_l) 
            if ttc < min_ttc:
                min_ttc = ttc
                target_obs = obs_tuple  
        return target_obs

    def calculate_to_lane_t_s(self):
        obs = self.select_target_obs()
        l_buffer = obs[1] - config_.ego_l
        if obs[4] != 0:
            t = l_buffer / obs[4]
        else:
            t = None  
        if t is not None:
            s = obs[3] * t + obs[0]
        else:
            s = None 
        return s, t
        
    def make_decision(self):
        obs      = self.select_target_obs()
        decision = []
        if obs == self.obs_list[0]:
            obs_class = 'point_obs'
        elif obs == self.obs_list[1]:
            obs_class = 'line_obs'

        if(obs_class == 'point_obs'):
            decision=['加速超车','减速让行','左绕行','左变道','右变道']
        elif(obs_class == 'line_obs'):
            decision=['减速让行','左绕行','左变道','右变道','匀速']
        # elif(obs == 'non_obs'):
        #     decision=['加速超车','减速让行','左绕行','右绕行','左变道','右变道']
        return decision
    
    #该函数计算绕行时的时间t
    def calculate_ref_t(self):
        ego_v_s             = config_.ego_v_s
        ego_s               = config_.ego_s
        obs                 = self.select_target_obs()
        b                   = ego_v_s - obs[3]
        c                   = ego_s -obs[0]
        delta               = b**2 - 2 * config_.acc_comfort * c

        if delta >= 0:
            t1 = (-b - math.sqrt(delta)) / (2 * config_.acc_comfort)
            t2 = (-b + math.sqrt(delta)) / (2 * config_.acc_comfort)

            if t1 > 0  :
                return t1
            elif t2 > 0:
                return t2

    def calculate_target_point(self):
        target_point                        = []
        obs                                 = self.select_target_obs()
        obs_to_lane_s,obs_to_lane_t         = self.calculate_to_lane_t_s()
        l_buffer                            = obs[1] - config_.ego_l
        decision                            = self.make_decision()    
        for dec in decision:
            if obs[4] != 0:
                if dec=='加速超车':
                    if(config_.ego_s+config_.ego_v_s*obs_to_lane_t+0.5*config_.acc_comfort*obs_to_lane_t**2 < obs_to_lane_s):
                        continue
                    #TODO 判断前车的逻辑，现在仿真环境没有这个上游信息
                    # elif(config_.ego_s+config_.ego_v_s*config_.t_total+0.5*config_.acc_max**2 > s_front+v_front*config_.t_total):
                    #     continue
                    elif(l_buffer < config_.l_safe):
                        continue
                    s_target            = config_.ego_s + config_.ego_v_s*config_.t_total + 0.5*config_.acc_comfort*config_.t_total**2
                    l_target            = config_.ego_l
                    t_target            = config_.t_total
                    target_array        = [s_target,l_target,t_target]
                    target_point.append(target_array)

            if dec=='减速让行':
                if(config_.ego_s+config_.ego_v_s*config_.t_total+0.5*config_.dcc_comfort*config_.t_total**2 > obs[3]*config_.t_total-2):
                    continue
                if(obs[0]-config_.ego_s > 30):
                    continue
                s_target            = config_.ego_s + config_.ego_v_s*config_.t_total + 0.5*config_.dcc_comfort*config_.t_total**2
                l_target            = config_.ego_l
                t_target            = config_.t_total
                target_array        = [s_target,l_target,t_target]
                target_point.append(target_array)

            if dec=='左绕行': #包含借道的逻辑
                in_left_max = False
                lift_obs    = True
                if(obs[0]-config_.ego_s > 30):
                    continue
                if(in_left_max):
                    continue
                if(lift_obs):
                    continue
                #TODO 目标车道障碍物信息，待补充
                # if(config_.ego_s-10 > targrt_lane_obs_s or targrt_lane_obs_s<obs[0] + 30):
                #     continue
                # TODO：少一个逻辑
                t_ref_1             = self.calculate_ref_t()
                v_ref_1             = config_.ego_v_s + config_.acc_comfort*t_ref_1
                s_ref_1             = obs[0] + obs[3] * t_ref_1
                # TODO：ref_2的逻辑还没整理好
                # t_ref_2           = 
                # v_ref_2           =
                s_ref_point_1       = s_ref_1
                l_ref_point_1       = obs[1] + 3 #TODO：后续改成和道路宽度相关
                t_ref_point_1       = t_ref_1
                # s_ref_point_2     = config_.ego_s + config_.ego_v_s*config_.t_total + 0.5*0.5*config_.dcc_comfort*config_.t_total**2
                # l_ref_point_2     = config_.ego_l
                # t_ref_point_2     = config_.t_total
                s_target            = config_.ego_s + config_.ego_v_s*config_.t_total + 0.5*config_.acc_comfort*config_.t_total**2
                l_target            = config_.ego_l
                t_target            = config_.t_total
                target_array        = [(s_ref_point_1,l_ref_point_1,t_ref_point_1),
                                       (s_target,l_target,t_target)]
                target_point.append(target_array)

            if dec=='左变道':
                in_left_max = False
                # if(obs[0]-config_.ego_s > 30):
                #     continue
                if(in_left_max):
                    continue
                # if(config_.ego_s-10 > targrt_lane_obs_s or targrt_lane_obs_s<obs_s+30):
                #     continue
                s_target            = config_.ego_s + config_.ego_v_s*config_.t_total + 0.5*config_.acc_comfort*config_.t_total**2
                l_target            = config_.ego_l + 4
                t_target            = config_.t_total
                target_array        = [s_target,l_target,t_target]
                target_point.append(target_array)

            if dec=='右变道':
                in_right_max = True
                if(obs[0]-config_.ego_s > 30):
                    continue
                if(in_right_max):
                    continue
                # if(config_.ego_s-10>targrt_lane_obs_s or targrt_lane_obs_s<obs_s+30):
                #     continue
                s_target            = config_.ego_s + config_.ego_v_s*config_.t_total + 0.5*config_.acc_comfort*config_.t_total**2
                l_target            = config_.ego_l -4
                t_target            = config_.t_total
                target_array        = [s_target,l_target,t_target]
                target_point.append(target_array)

            if dec=='匀速':
                Slow_down_the_front_car     = False
                ego_v_is_greater_than_obs   = False
                if(Slow_down_the_front_car or ego_v_is_greater_than_obs):
                    continue
                s_target            = config_.ego_s + config_.ego_v_s*config_.t_total
                l_target            = config_.ego_l
                t_target            = config_.t_total
                target_array        = [s_target,l_target,t_target]
                target_point.append(target_array)

        return target_point


class DWA:
    def __init__(self):
        # 初始化参数
        self.update_ = Update()
        self.dt = 0.27
        self.max_speed = 15
        self.min_speed = 0
        self.max_acceleration = 3
        self.final_target_point = [70, 2]
        self.max_yawrate = 40.0 * math.pi / 180.0
        self.max_accel = 3
        self.max_dyawrate = 40.0 * math.pi / 180.0
        self.v_reso = 0.1
        self.yawrate_reso = 0.1 * math.pi / 180.0
        self.predict_time = 3.0
        self.to_goal_cost_gain = 1.0
        self.robot_radius = 1.0
        self.trajectory_points = []  # 存储轨迹点
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def motion(self,x, u, dt):
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[2] += u[1] * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_dynamic_window(self,x):
        vs = [self.min_speed, 0.4*self.max_speed, -self.max_yawrate, self.max_yawrate]
        vd = [x[3] - self.max_accel * self.dt, x[3] + self.max_accel * self.dt,
              x[4] - self.max_dyawrate * self.dt, x[4] + self.max_dyawrate * self.dt]
        vr = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]
        return vr

    def calc_trajectory(self, x_init, v, w):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, w], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt
        return trajectory

    def calc_to_goal_cost(self, trajectory, current_goal_index, goal_points):
        # 计算到当前目标点的代价
        goal = goal_points[current_goal_index]
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        goal_dis = math.sqrt(dx ** 2 + dy ** 2)
        cost = self.to_goal_cost_gain * goal_dis
        return cost

    def compute_obstacle_cost(self, position, obstacles):
        cost = 0.0
        for obstacle in obstacles:
            distance = np.linalg.norm(position - obstacle)
            cost += 1 / (distance + 1e-6)
        return cost

    def calc_obstacle_cost(self, traj, ob):
        min_r = float("inf")
        for ii in range(len(traj)):
            for i in range(len(ob)):
                ox = ob[i, 0]
                oy = ob[i, 1]
                dx = traj[ii, 0] - ox
                dy = traj[ii, 1] - oy
                r = math.sqrt(dx ** 2 + dy ** 2)
                if r <= self.robot_radius:
                    return float("Inf")
                if min_r >= r:
                    min_r = r
        return 1.0 / min_r

    def calc_final_input(self, x, u, vr, current_goal_index, goal_points, ob):
        x_init = x[:]
        min_cost = float("inf")
        min_u = u
        best_trajectory = np.array([x])

        for v in np.arange(vr[0], vr[1], self.v_reso):
            for w in np.arange(vr[2], vr[3], self.yawrate_reso):
                trajectory = self.calc_trajectory(x_init, v, w)
                to_goal_cost = self.calc_to_goal_cost(trajectory, current_goal_index, goal_points)
                ref_cost = abs(x[1] - 2)
                speed_cost = abs(7 - x[4])
                ob_cost = self.calc_obstacle_cost(trajectory, ob)
                final_cost = to_goal_cost + speed_cost + ob_cost + 10 * ref_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    min_u = [v, w]
                    best_trajectory = trajectory

        return min_u, best_trajectory
    
    def dwa_control(self, x, u, goal_points, ob):
        vr = self.calc_dynamic_window(x)
        current_goal_index = self.current_goal_index  # 选择当前的目标点索引
        u, trajectory = self.calc_final_input(x, u, vr, current_goal_index, goal_points, ob)
        return u, trajectory

    def _create_box(self, length, width, height, s, l, t):
        l_half = length / 2
        w_half = width / 2
        h_half = height / 2

        x = [l_half, l_half, -l_half, -l_half, l_half, l_half, -l_half, -l_half]
        y = [w_half, -w_half, -w_half, w_half, w_half, -w_half, -w_half, w_half]
        z = [h_half, h_half, h_half, h_half, -h_half, -h_half, -h_half, -h_half]

        x = [xi + s for xi in x]
        y = [yi + l for yi in y]
        z = [zi + t for zi in z]

        return np.array(list(zip(x, y, z)))

    def plot_arrow(self, ax, x, y, z, yaw, length=0.5, width=0.1):
        ax.quiver(x, y, z, length * math.cos(yaw), length * math.sin(yaw), 0, 
                  arrow_length_ratio=0.3)
    
    def plot_rectangle(self, ax, s, l, t, color='red', length=2, width=1, height=0):
        ax.bar3d(s, l, t, length, width, height, color=color, alpha=0.5)

    def draw_path(self, trajectory, goal, ob, x):
        self.ax.plot(ob[:, 0], ob[:, 1], ob[:, 2], "bs")
        self.ax.set_xlabel('s')
        self.ax.set_ylabel('l')
        self.ax.set_zlabel('t')
        self.ax.set_xlim([0, 150])
        self.ax.set_ylim([0, 8])
        self.ax.set_zlim([0, 10])
        self.ax.set_box_aspect([8, 1.5, 2])
        self.ax.grid(True)
        plt.show()

    def draw_dynamic_search(self, target_point,best_trajectory,cur_t, x, goal, ob):
        self.ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], best_trajectory[:, 4], "-g")
        self.ax.plot([x[0]], [x[1]], cur_t, "xr")
        self.ax.plot(ob[:, 0], ob[:, 1], ob[:, 2], "bs")
        all_target_points = []

        for element in target_point:
            if isinstance(element, (list, tuple)):
                if len(element) == 3:
                    all_target_points.append(element)
                else:
                    for item in element:
                        if isinstance(item, (list, tuple)) and len(item) == 3:
                            all_target_points.append(item)

        target_points_list = [list(point) for point in all_target_points]
        target_points_array = np.array(target_points_list)

        self.ax.scatter(target_points_array[:, 0], target_points_array[:, 1], target_points_array[:, 2],
                color='r', label='Target Points')        
        # Draw rectangles for best trajectory and obstacles
        # for point in best_trajectory:
        #     self.plot_rectangle(self.ax, point[0], point[1], cur_t)
        for obs in ob:
            self.plot_rectangle(self.ax, obs[0], obs[1], obs[2], color='blue')
        
        # Additional settings
        self.ax.set_xlabel('s')
        self.ax.set_ylabel('l')
        self.ax.set_zlabel('t')
        self.ax.set_xlim([0, 150])
        self.ax.set_ylim([0, 8])
        self.ax.set_zlim([0, 10])
        self.ax.set_box_aspect([8, 1.5, 2])
        self.ax.grid(True)
        plt.pause(0.0001)

    def update_plot(self, i, trajectory, goal, ob):
        x = trajectory[i]
        # Plot the trajectory and obstacles
        self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 4], "-r")
        self.ax.plot(ob[:, 0], ob[:, 1], ob[:, 2], "bs")
        self.ax.plot([x[0]], [x[1]], [x[4]], "xr")
        
        # # Draw rectangles for trajectory and obstacles
        # for point in trajectory:
        #     self.plot_rectangle(self.ax, point[0], point[1], point[4])
        # for obs in ob:
        #     self.plot_rectangle(self.ax, obs[0], obs[1], obs[2], color='blue')
        
        # Additional settings
        self.ax.set_xlabel('s')
        self.ax.set_ylabel('l')
        self.ax.set_zlabel('t')
        self.ax.set_xlim([0, 150])
        self.ax.set_ylim([0, 8])
        self.ax.set_zlim([0, 10])
        self.ax.set_box_aspect([8, 1.5, 2])
        self.ax.grid(True)

    def animate(self, trajectory, goal, ob, interval=100):
        ani = FuncAnimation(self.fig, self.update_plot, frames=len(trajectory), fargs=(trajectory, goal, ob), interval=interval)
        plt.show()

    def main(self):
        x = np.array([0.0, 2.0, 0.0, 7.0, 0.0])
        goal_points = np.array([[120, 6]])  # 目标点集合
        update = Update()
        u = np.array([4, 0.0])
        trajectory = np.array([x])
        self.trajectory_points = [] 
        target_point = rulebase.calculate_target_point()
        print(target_point)

        self.current_goal_index = 0  # 添加当前目标点索引
        while self.current_goal_index < len(goal_points):
            update.update()
            ob = update.get_obstacle_positions()
            u, best_trajectory = self.dwa_control(x, u, goal_points, ob)  # 修改调用以传递所有参数
            x = self.motion(x, u, self.dt)
            cur_t = update.current_t
            self.draw_dynamic_search(target_point, best_trajectory, cur_t, x, goal_points, ob)
            trajectory = np.vstack((trajectory, x))

            # 检查是否到达当前目标点
            if np.linalg.norm(x[:2] - goal_points[self.current_goal_index]) <= self.robot_radius:
                print(f"Reached goal point {self.current_goal_index}!")
                self.current_goal_index += 1

            if self.current_goal_index >= len(goal_points):
                print("All goals reached!")
                break

        print("Done")
        self.draw_path(trajectory, goal_points, ob, x)
        self.animate(trajectory, goal_points, ob)
config_         = Config()
update_         = Update()
rulebase        = Rulebase()

update_.config  = config_
update_.reset()
dwa = DWA()

dwa.main()
# update_.simulate(100)
