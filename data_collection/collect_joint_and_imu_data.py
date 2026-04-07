"""
Collect Joint and IMU Data During Autowalk (using streaming API).

This script monitors for an autowalk mission started from the tablet controller
and records joint and IMU data while the mission is running.

Uses GetRobotStateStream for high-frequency, reliable data collection (~333 Hz)
instead of polling GetRobotState.

Two CSV files are written per experiment since joints and IMU run at different rates:
  - <experiment_name>_joints_<timestamp>.csv  (~333 Hz)
  - <experiment_name>_imu_<timestamp>.csv     (~1000 Hz, 3 packets per stream message)

Data is written incrementally (flushed every 100 joint samples) to prevent
data loss if the script is interrupted.

Usage:
    python3 collect_joint_and_imu_data.py 192.168.80.3 --experiment_name baseline_loop3 --output_dir /home/nerve/Desktop/data_collected/flat_Mar_20


The script will:
1. Wait for an autowalk mission to start (from tablet)
2. Begin recording joint + IMU data when mission starts
3. Stop recording when mission completes
4. Save data to: <experiment_name>_joints_YYYYMMDD_HHMMSS.csv
                 <experiment_name>_imu_YYYYMMDD_HHMMSS.csv
"""

import argparse
import csv
import os
import threading
import time
from datetime import datetime

import bosdyn.client
import bosdyn.client.util
import bosdyn.mission.client
from bosdyn.api.mission import mission_pb2
from bosdyn.client.robot_state import RobotStateStreamingClient

# Joint names in order (matching streaming API array indices)
# Order defined by spot_constants_pb2.JOINT_INDEX_*
JOINT_NAMES = [
    "fl.hx", "fl.hy", "fl.kn",  # front-left: hip x, hip y, knee
    "fr.hx", "fr.hy", "fr.kn",  # front-right
    "hl.hx", "hl.hy", "hl.kn",  # hind-left
    "hr.hx", "hr.hy", "hr.kn",  # hind-right
]

JOINT_CSV_HEADER = (
    # ['timestamp', 'elapsed_time']
    [f'{j}_position' for j in JOINT_NAMES]
    + [f'{j}_velocity' for j in JOINT_NAMES]
    + [f'{j}_load'     for j in JOINT_NAMES]
    # Body pose in vision frame (vision_tform_body)
    + ['vision_tform_body_pos_x', 'vision_tform_body_pos_y', 'vision_tform_body_pos_z']
    + ['vision_tform_body_rot_x', 'vision_tform_body_rot_y', 'vision_tform_body_rot_z', 'vision_tform_body_rot_w']
    # Body velocity in vision frame (velocity_of_body_in_vision)
    + ['vel_of_body_in_vision_lin_x', 'vel_of_body_in_vision_lin_y', 'vel_of_body_in_vision_lin_z']
    + ['vel_of_body_in_vision_ang_x', 'vel_of_body_in_vision_ang_y', 'vel_of_body_in_vision_ang_z']
    # Foot contact states (contact_states), ordered FL, FR, HL, HR
    + ['contact_fl', 'contact_fr', 'contact_hl', 'contact_hr']
)

IMU_CSV_HEADER = [
    # 'timestamp',
    'acc_x', 'acc_y', 'acc_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'quat_x', 'quat_y', 'quat_z', 'quat_w',
]


def main():
    parser = argparse.ArgumentParser(
        description='Collect joint + IMU data while autowalk is running (started from tablet)')

    bosdyn.client.util.add_base_arguments(parser)

    parser.add_argument('--experiment_name', required=True,
                        help='Name of your experiment')
    parser.add_argument('--output_dir', default='.',
                        help='Directory to save CSV files')

    args = parser.parse_args()

    if args.output_dir != '.' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f'Created output directory: {args.output_dir}')

    sdk = bosdyn.client.create_standard_sdk('JointIMUCollectorDuringAutowalk',
                                             [bosdyn.mission.client.MissionClient])
    sdk.register_service_client(RobotStateStreamingClient)

    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync(timeout_sec=10.0)

    robot_state_streaming_client = robot.ensure_client(RobotStateStreamingClient.default_service_name)
    mission_client = robot.ensure_client(bosdyn.mission.client.MissionClient.default_service_name)

    print(f'Connected to robot at {args.hostname}')
    print(f'Experiment: {args.experiment_name}')
    print('Using streaming API for high-frequency data collection')
    print('Waiting for autowalk mission to start (press Ctrl+C to exit)...\n')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    try:
        while True:
            mission_state = mission_client.get_state()
            if mission_state.status != mission_pb2.State.STATUS_RUNNING:
                time.sleep(0.5)
                continue

            print('Autowalk started!')

            joints_file = os.path.join(args.output_dir, f'{args.experiment_name}_joints_{timestamp_str}.csv')
            imu_file    = os.path.join(args.output_dir, f'{args.experiment_name}_imu_{timestamp_str}.csv')

            joint_count = 0
            imu_count   = 0
            start_time  = None
            stop_recording = threading.Event()

            def monitor_mission():
                while not stop_recording.is_set():
                    try:
                        state = mission_client.get_state()
                        if state.status != mission_pb2.State.STATUS_RUNNING:
                            stop_recording.set()
                            return
                    except:
                        pass
                    time.sleep(0.5)

            monitor_thread = threading.Thread(target=monitor_mission, daemon=True)
            monitor_thread.start()

            with open(joints_file, 'w', newline='') as jf, \
                 open(imu_file,    'w', newline='') as imuf:

                joint_writer = csv.writer(jf)
                imu_writer   = csv.writer(imuf)

                joint_writer.writerow(JOINT_CSV_HEADER)
                imu_writer.writerow(IMU_CSV_HEADER)

                try:
                    for state in robot_state_streaming_client.get_robot_state_stream():
                        if stop_recording.is_set():
                            break

                        # now = time.time()
                        # if start_time is None:
                        #     start_time = now

                        # elapsed = now - start_time

                        # --- Joint row (~333 Hz) ---
                        positions  = list(state.joint_states.position)
                        velocities = list(state.joint_states.velocity)
                        loads      = list(state.joint_states.load)

                        kin = state.kinematic_state
                        pos  = kin.vision_tform_body.position
                        rot  = kin.vision_tform_body.rotation
                        vlin = kin.velocity_of_body_in_vision.linear
                        vang = kin.velocity_of_body_in_vision.angular

                        # row = [f'{now:.9f}', f'{elapsed:.6f}']
                        row = [f'{v:.6f}' for v in positions]
                        row += [f'{v:.6f}' for v in velocities]
                        row += [f'{v:.6f}' for v in loads]
                        row += [f'{pos.x:.6f}',  f'{pos.y:.6f}',  f'{pos.z:.6f}']
                        row += [f'{rot.x:.6f}',  f'{rot.y:.6f}',  f'{rot.z:.6f}',  f'{rot.w:.6f}']
                        row += [f'{vlin.x:.6f}', f'{vlin.y:.6f}', f'{vlin.z:.6f}']
                        row += [f'{vang.x:.6f}', f'{vang.y:.6f}', f'{vang.z:.6f}']
                        row += [int(c) for c in state.contact_states]
                        joint_writer.writerow(row)
                        joint_count += 1

                        # --- IMU rows (~1000 Hz, 3 packets per message) ---
                        for packet in state.inertial_state.packets:
                            # t = packet.timestamp.seconds + packet.timestamp.nanos * 1e-9
                            # elapsed_imu = t
                            acc  = packet.acceleration_rt_odom_in_link_frame
                            gyro = packet.angular_velocity_rt_odom_in_link_frame
                            quat = packet.odom_rot_link

                            imu_row = [
                                # f'{t:.9f}',
                                f'{acc.x:.6f}',  f'{acc.y:.6f}',  f'{acc.z:.6f}',
                                f'{gyro.x:.6f}', f'{gyro.y:.6f}', f'{gyro.z:.6f}',
                                f'{quat.x:.6f}', f'{quat.y:.6f}', f'{quat.z:.6f}', f'{quat.w:.6f}',
                            ]
                            imu_writer.writerow(imu_row)
                            imu_count += 1

                        # Flush every 100 joint samples (~0.3 s)
                        if joint_count % 1000 == 0:
                            jf.flush()
                            imuf.flush()
                            # print(f'{joint_count} joint samples | {imu_count} IMU samples collected...')

                except Exception as e:
                    print(f'Stream error: {e}')

            print(f'Autowalk ended!')
            print(f'  Joint samples : {joint_count}  → {joints_file}')
            print(f'  IMU samples   : {imu_count}   → {imu_file}')

    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
