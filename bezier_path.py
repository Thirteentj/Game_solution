import matplotlib.pyplot as plt
import numpy as np
import scipy.special

show_animation = True


def calc_4points_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset):
    """
    Compute control points and path given start and end position.
    :param sx: (float) x-coordinate of the starting point
    :param sy: (float) y-coordinate of the starting point
    :param syaw: (float) yaw angle at start
    :param ex: (float) x-coordinate of the ending point
    :param ey: (float) y-coordinate of the ending point
    :param eyaw: (float) yaw angle at the end
    :param offset: (float)
    :return: (numpy array, numpy array)
    """
    dist = np.hypot(sx - ex, sy - ey) / offset
    control_points = np.array(
        [[sx, sy],
         [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
         [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
         [ex, ey]])

    path = calc_bezier_path(control_points, n_points=100)

    return path, control_points


def calc_bezier_path(control_points, n_points=100):
    """
    Compute bezier path (trajectory) given control points.
    :param control_points: (numpy array)
    :param n_points: (int) number of points in the trajectory
    :return: (numpy array)
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def bernstein_poly(n, i, t):
    """
    Bernstein polynom.
    :param n: (int) polynom degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier(t, control_points):
    """
    Return one point on the bezier curve.
    :param t: (float) number in [0, 1]
    :param control_points: (numpy array)
    :return: (numpy array) Coordinates of the point
    """
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)


def bezier_derivatives_control_points(control_points, n_derivatives):
    """
    Compute control points of the successive derivatives of a given bezier curve.
    A derivative of a bezier curve is a bezier curve.
    See https://pomax.github.io/bezierinfo/#derivatives
    for detailed explanations
    :param control_points: (numpy array)
    :param n_derivatives: (int)
    e.g., n_derivatives=2 -> compute control points for first and second derivatives
    :return: ([numpy array])
    """
    w = {0: control_points}
    for i in range(n_derivatives):
        n = len(w[i])
        w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j])
                             for j in range(n - 1)])
    return w


def curvature(dx, dy, ddx, ddy):
    """
    Compute curvature at one point given first and second derivatives.
    :param dx: (float) First derivative along x axis
    :param dy: (float)
    :param ddx: (float) Second derivative along x axis
    :param ddy: (float)
    :return: (float)
    """
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """Plot arrow."""
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():
    """Plot an example bezier curve."""
    start_x = 10.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.radians(180.0)  # [rad]

    end_x = -0.0  # [m]
    end_y = -10.0  # [m]
    end_yaw = np.radians(270.0)  # [rad]
    offset = 3.0

    path, control_points = calc_4points_bezier_path(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)
    # print('path')
    # print(path)
    # Note: alternatively, instead of specifying start and end position
    # you can directly define n control points and compute the path:
    # control_points = np.array([[5., 1.], [-2.78, 1.], [-11.5, -4.5], [-6., -8.]])
    # path = calc_bezier_path(control_points, n_points=100)

    # Display the tangent, normal and radius of cruvature at a given point
    t = 0.86  # Number in [0, 1]
    x_target, y_target = bezier(t, control_points)
    derivatives_cp = bezier_derivatives_control_points(control_points, 2)
    point = bezier(t, control_points)
    dt = bezier(t, derivatives_cp[1])
    ddt = bezier(t, derivatives_cp[2])
    # Radius of curvature
    radius = 1 / curvature(dt[0], dt[1], ddt[0], ddt[1])
    # Normalize derivative
    dt /= np.linalg.norm(dt, 2)
    tangent = np.array([point, point + dt])
    normal = np.array([point, point + [- dt[1], dt[0]]])
    curvature_center = point + np.array([- dt[1], dt[0]]) * radius
    circle = plt.Circle(tuple(curvature_center), radius,
                        color=(0, 0.8, 0.8), fill=False, linewidth=1)

    assert path.T[0][0] == start_x, "path is invalid"
    assert path.T[1][0] == start_y, "path is invalid"
    assert path.T[0][-1] == end_x, "path is invalid"
    assert path.T[1][-1] == end_y, "path is invalid"

    if show_animation:  # pragma: no cover
        fig, ax = plt.subplots()
        ax.plot(path.T[0], path.T[1], label="Bezier Path")
        ax.plot(control_points.T[0], control_points.T[1],
                '--o', label="Control Points")
        ax.plot(x_target, y_target)
        ax.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
        ax.plot(normal[:, 0], normal[:, 1], label="Normal")
        ax.add_artist(circle)
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        plt.show()

    return path, control_points


def main2():
    """Show the effect of the offset."""
    start_x = 10.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.radians(180.0)  # [rad]

    end_x = -0.0  # [m]
    end_y = -10.0  # [m]
    end_yaw = np.radians(270.0)  # [rad]

    for offset in np.arange(1.0, 5.0, 1.0):
        path, control_points = calc_4points_bezier_path(
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)
        assert path.T[0][0] == start_x, "path is invalid"
        assert path.T[1][0] == start_y, "path is invalid"
        assert path.T[0][-1] == end_x, "path is invalid"
        assert path.T[1][-1] == end_y, "path is invalid"

        if show_animation:  # pragma: no cover
            plt.plot(path.T[0], path.T[1], label="Offset=" + str(offset))

    if show_animation:  # pragma: no cover
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def get_planning_path_bezier(start_point,start_yaw,end_point,end_yaw):
    """Plot an example bezier curve."""
    start_x,start_y = start_point[0],start_point[1]  # [m]
    start_yaw = start_yaw  # [rad]

    end_x,end_y = end_point[0],end_point[1]  # [m]
    end_yaw = end_yaw  # [rad]
    offset = 3.0

    path, control_points = calc_4points_bezier_path(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)


    t = 0.86  # Number in [0, 1]
    x_target, y_target = bezier(t, control_points)
    derivatives_cp = bezier_derivatives_control_points(control_points, 2)
    point = bezier(t, control_points)
    dt = bezier(t, derivatives_cp[1])
    ddt = bezier(t, derivatives_cp[2])
    # Radius of curvature
    radius = 1 / curvature(dt[0], dt[1], ddt[0], ddt[1])
    # Normalize derivative
    dt /= np.linalg.norm(dt, 2)
    tangent = np.array([point, point + dt])
    normal = np.array([point, point + [- dt[1], dt[0]]])
    curvature_center = point + np.array([- dt[1], dt[0]]) * radius
    circle = plt.Circle(tuple(curvature_center), radius,
                        color=(0, 0.8, 0.8), fill=False, linewidth=1)

    assert path.T[0][0] == start_x, "path is invalid"
    assert path.T[1][0] == start_y, "path is invalid"
    assert path.T[0][-1] == end_x, "path is invalid"
    assert path.T[1][-1] == end_y, "path is invalid"

    # if show_animation:  # pragma: no cover
    #     fig, ax = plt.subplots()
    #     ax.plot(path.T[0], path.T[1], label="Bezier Path")
    #     ax.plot(control_points.T[0], control_points.T[1],
    #             '--o', label="Control Points")
    #     ax.plot(x_target, y_target)
    #     ax.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
    #     ax.plot(normal[:, 0], normal[:, 1], label="Normal")
    #     ax.add_artist(circle)
    #     plot_arrow(start_x, start_y, start_yaw)
    #     plot_arrow(end_x, end_y, end_yaw)
    #     ax.legend()
    #     ax.axis("equal")
    #     ax.grid(True)
    #     plt.show()

    return path

def location_xy2_s(path_xy):  #xy坐标系转换到自然坐标系
    #已知二维坐标，转换为自然坐标

    path_len = [0]  #自然坐标系下的距离存储
    path_dire = [0] #角度存储
    path_s = []
    for i in range(len(path_xy[0]) - 1):
        point_x,point_y = path_xy[0][i],path_xy[1][i]
        point_x_next, point_y_next = path_xy[0][i+1], path_xy[1][i+1]
        dx = point_x_next-point_x
        dy = point_y_next-point_y
        dis_interval = np.sqrt((dx)**2+(dy)**2)

        direction = lambda d: d > 0 and d or d + 2 * np.pi, np.arctan2(dy, dx)  #计算转向角
        # len_sum = dis_interval + path_len[-1]  #计算累积长度
        # path_len.append(len_sum)   #这里暂时不记录累积距离，记录每个分段的距离
        path_len.append(dis_interval)
        path_dire.append(direction[1])
    path_s.append(path_len)
    path_s.append(path_dire)

    return path_s

def path_mat_trans(path):
    # 坐标存储格式转换
    path_new = []
    path_x = [point[0] for point in path]
    path_y = [point[1] for point in path]
    path_new.append(path_x)
    path_new.append(path_y)
    return path_new

def location_s2_xy(start_point,path_s):   #自然坐标系转换到xy坐标系 已知自然坐标，转换到二维坐标
    path_len = path_s[0]  #自然坐标系下的距离存储
    path_dire = path_s[1]  #角度存储
    dx = path_len * np.cos(path_dire)
    dy = path_len * np.sin(path_dire)
    print('dx',dx,dy)
    x_point = []
    y_point = []
    xy_point = []
    for i in range(len(path_len)):
        if i == 0:
            dx_s = start_point[0] + dx[i]
            dy_s = start_point[1] + dy[i]
        else:
            dx_s = x_point[-1]+dx[i]
            dy_s = y_point[-1]+dy[i]
        x_point.append(dx_s)
        y_point.append(dy_s)
    xy_point.append(x_point)
    xy_point.append(y_point)
    return xy_point

if __name__ == '__main__':

    path, control_points = main()
    #  main2()
    print('path',path)
    path = path_mat_trans(path)
    path_s = location_xy2_s(path)   #自然坐标系中记录的距离为每段的分段间距
    print('sss')
    print(path_s)
    path_xy = location_s2_xy((10,1),path_s)
    print(path_xy)
