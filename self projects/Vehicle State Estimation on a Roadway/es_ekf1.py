import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion
from numpy.linalg import inv
import rotations


# for sensor dropout 
# comment following two lines
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

#uncomment following two lines
# with open('data/pt3_data.pkl', 'rb') as file:
#     data = pickle.load(file)


# for sensor miscalibration
# comment following C_li
C_li = np.array([
    [ 0.99376, -0.09722,  0.05466],
    [ 0.09971,  0.99401, -0.04475],
    [-0.04998,  0.04992,  0.9975 ]])

# uncomment the following C_li
# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
# C_li = np.array([
#       [ 0.9975 , -0.04742,  0.05235],
#       [ 0.04992,  0.99763, -0.04742],
#       [-0.04998,  0.04992,  0.9975 ]])


gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']


gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()



# translation vector
t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li


var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 10.00 # default: 0.01
                  # good choice for part 3: 10.00
var_lidar = 10.00  # default: 1.00
                  # good choice for part 2: 100.00
                  # good choice for part 3: 10.00


g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian


p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion( euler = gt.r[0] ).to_numpy()
print("q_est[0] =", q_est[0])
C_ns_0 = Quaternion(*q_est[0]).to_mat()
print("C_ns_0 =", C_ns_0)
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0




def skew_operator(a):
    op_mat = np.zeros([3, 3])
    op_mat[0, 1] = -a[2]
    op_mat[0, 2] = a[1]
    op_mat[1, 0] = a[2]
    op_mat[1, 2] = -a[0]
    op_mat[2, 0] = -a[1]
    op_mat[2, 1] = a[0]

    return op_mat



def quaternion_left_prod(theta):

    ### normalize the angle first
    theta = angle_normalize(theta)

    ### construct quaternion based on theta info
    theta_norm = np.sqrt(theta[0] ** 2 + theta[1] ** 2 + theta[2] ** 2)
    q_w = np.sin(theta_norm / 2)
    q_v = theta / theta_norm * np.cos(theta_norm / 2)

    Omega = np.zeros([4, 4])
    Omega[0, 1:4]   = -q_v.T
    Omega[1:4, 0]   = q_v
    Omega[1:4, 1:4] = skew_operator(q_v)
    Omega = Omega + np.identity(4) * q_w

    return Omega


def quaternion_right_prod(theta):

    ### normalize the angle first
    theta = angle_normalize(theta)

    ### construct quaternion based on theta info
    theta_norm = np.sqrt(theta[0] ** 2 + theta[1] ** 2 + theta[2] ** 2)
    q_w = np.sin(theta_norm / 2)
    q_v = theta / theta_norm * np.cos(theta_norm / 2)

    Omega = np.zeros([4, 4])
    Omega[0, 1:4]   = -q_v.T
    Omega[1:4, 0]   = q_v
    Omega[1:4, 1:4] = -skew_operator(q_v)
    Omega = Omega + np.identity(4) * q_w

    return Omega

def normalize_quaternion(q):

    norm = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)

    return q / norm


def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):


    H_k = np.zeros([3, 9])
    H_k[0:3, 0:3] = np.identity(3)

    K_k = p_cov_check @ H_k.T @ inv(H_k @ p_cov_check @ H_k.T + sensor_var)


    delta_x_k = K_k @ (y_k - p_check)
    p_hat = p_check + delta_x_k[0:3]
    v_hat = v_check + delta_x_k[3:6]

    q_obj = Quaternion( euler = delta_x_k[6:9] ).quat_mult_left(q_check)
    q_hat = Quaternion(*q_obj).normalize().to_numpy()
    # q_hat = Quaternion(*q_obj).to_numpy() # Note: after test, it tuns out we don't have to normalize the quaternion


    p_cov_hat = ( np.identity(9) - K_k @ H_k ) @ p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat



# Define some suppotive variables
R_GNSS      =  np.identity(3) * var_gnss    # covariance matrix related to GNSS
R_Lidar     =  np.identity(3) * var_lidar   # covariance matrix related to Lidar
t_imu       =  imu_f.t                      # timestanps of imu
t_gnss      =  gnss.t                       # timestamps of gnss
t_lidar     =  lidar.t                      # timestamps of lidar 
F_k         =  np.identity(9)
L_k         =  np.zeros([9, 6])
L_k[3:9, :] =  np.identity(6)
Q           =  np.identity(6)               # covariance matrix related to noise of IMU
Q[0:3, 0:3] =  Q[0:3, 0:3] * var_imu_f      # covariance matrix related to special force of IMU
Q[3:6, 3:6] =  Q[3:6, 3:6] * var_imu_w      

for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt

    #
    delta_t = imu_f.t[k] - imu_f.t[k - 1]
    Q_k = Q * delta_t * delta_t
    C_ns = Quaternion(*q_est[k-1]).to_mat()
    # print("C_ns = ", C_ns)

    p_est[k] = p_est[k-1] + delta_t * v_est[k-1] + delta_t ** 2 / 2 * (C_ns @ imu_f.data[k-1] + g)
    v_est[k] = v_est[k-1] + delta_t * (C_ns @ imu_f.data[k-1] + g)

    q_tmp = Quaternion( euler = (imu_w.data[k-1] * delta_t) ).quat_mult_right( q_est[k-1] )
    q_est[k] = Quaternion(*q_tmp).normalize().to_numpy()

    F_k[0:3, 3:6] = np.identity(3) * delta_t
    F_k[3:6, 6:9] = - skew_operator( C_ns @ imu_f.data[k-1] ) * delta_t


    p_cov[k] = F_k @ p_cov[k-1] @ F_k.T + L_k @ Q_k @ L_k.T


    if np.any( t_gnss == t_imu[k] ):

        t_k = np.where( t_gnss == t_imu[k] )[0][0]
        [ p_est[k], v_est[k], q_est[k], p_cov[k] ] = measurement_update( R_GNSS, p_cov[k], gnss.data[t_k], p_est[k], v_est[k], q_est[k] )


    if np.any( t_lidar == t_imu[k] ):

        t_k = np.where( t_lidar == t_imu[k] )[0][0]
        [ p_est[k], v_est[k], q_est[k], p_cov[k] ] = measurement_update( R_Lidar, p_cov[k], lidar.data[t_k], p_est[k], v_est[k], q_est[k] )

    

#### Results and Analysis ###################################################################


est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()


error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

