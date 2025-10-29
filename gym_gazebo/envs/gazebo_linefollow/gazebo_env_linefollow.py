
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected
        
        # Bug Fix 1: Use subscriber callback instead of polling
        self.image_data = None
        self.image_sub = rospy.Subscriber('/pi_camera/image_raw', Image, self.image_callback)
    
    def image_callback(self, data):
        '''
        @brief Callback function for image subscriber
        @param data: Image data from ROS topic
        '''
        self.image_data = data


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        NUM_BINS = 3
        # State now includes: [current_position (10 bins), future_position (10 bins)]
        state = [0] * 20  # 10 bins for current + 10 bins for future
        done = False

        # Convert image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold to detect the line (assuming dark line on light background)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Get image dimensions
        height, width = binary.shape
        
        # Divide image into thirds
        roi_height = height // 3
        
        # Analyze BOTTOM third for current line position
        roi_bottom = binary[2 * roi_height:height, :]
        
        # Analyze MIDDLE third for future line position (look-ahead)
        roi_middle = binary[roi_height:2 * roi_height, :]
        
        # Calculate moments to find the centroid of the CURRENT line (bottom third)
        moments_current = cv2.moments(roi_bottom, binaryImage=True)
        
        # Calculate moments to find the centroid of the FUTURE line (middle third)
        moments_future = cv2.moments(roi_middle, binaryImage=True)
        
        # Create visualization image
        vis_image = cv_image.copy()
        
        # Draw ROI rectangles
        cv2.rectangle(vis_image, (0, 2 * roi_height), (width, height), (0, 255, 0), 2)  # Bottom ROI (current)
        cv2.rectangle(vis_image, (0, roi_height), (width, 2 * roi_height), (255, 0, 0), 2)  # Middle ROI (future)
        
        # Draw vertical lines for bins
        bin_width = width / 10
        for i in range(11):
            x = int(i * bin_width)
            color = (100, 100, 100)
            if i == 4 or i == 6:  # Highlight center bins
                color = (0, 255, 0)
            cv2.line(vis_image, (x, 0), (x, height), color, 1)
        
        current_bin = -1
        future_bin = -1
        
        # Process CURRENT line position (bottom third ROI)
        if moments_current['m00'] > 0:
            # Calculate centroid x-position
            cx = int(moments_current['m10'] / moments_current['m00'])
            
            # Draw the detected line center position (in bottom third)
            cy = 2 * roi_height + roi_height // 2
            cv2.circle(vis_image, (cx, cy), 10, (0, 255, 0), -1)
            
            # Determine which bin the line center falls into (0-9)
            current_bin = int(cx / bin_width)
            
            # Ensure bin_index is within valid range
            current_bin = max(0, min(9, current_bin))
            
            # Set the state array (first 10 elements for current position)
            state[current_bin] = 1
            
            # Highlight the active bin in the bottom third
            bin_left = int(current_bin * bin_width)
            bin_right = int((current_bin + 1) * bin_width)
            cv2.rectangle(vis_image, (bin_left, height - 60), (bin_right, height - 10), (0, 255, 0), -1)
            
            # Reset timeout since line was detected
            self.timeout = 0
        else:
            # No current line detected
            self.timeout += 1
        
        # Process FUTURE line position (middle third ROI)
        if moments_future['m00'] > 0:
            # Calculate centroid x-position for future line
            fx = int(moments_future['m10'] / moments_future['m00'])
            
            # Draw the detected future line center position (in middle third)
            fy = roi_height + roi_height // 2
            cv2.circle(vis_image, (fx, fy), 10, (255, 0, 0), -1)
            
            # Determine which bin the future line center falls into (0-9)
            future_bin = int(fx / bin_width)
            
            # Ensure bin_index is within valid range
            future_bin = max(0, min(9, future_bin))
            
            # Set the state array (last 10 elements for future position)
            state[10 + future_bin] = 1
            
            # Highlight the future active bin in the middle third
            bin_left = int(future_bin * bin_width)
            bin_right = int((future_bin + 1) * bin_width)
            cv2.rectangle(vis_image, (bin_left, roi_height + 10), (bin_right, roi_height + 60), (255, 0, 0), -1)
        
        # Add text showing bin numbers
        if current_bin >= 0:
            cv2.putText(vis_image, f"Current Bin: {current_bin}", (10, height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_image, "NO CURRENT LINE", (10, height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if future_bin >= 0:
            cv2.putText(vis_image, f"Future Bin: {future_bin}", (10, roi_height + roi_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(vis_image, "NO FUTURE LINE", (10, roi_height + roi_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
        
        if self.timeout > 0:
            cv2.putText(vis_image, f"Timeout: {self.timeout}/30", (10, height - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check if timeout threshold exceeded
        if self.timeout > 30:
            done = True

        # Display the visualization
        cv2.imshow("Robot Camera View", vis_image)
        cv2.imshow("Binary Threshold", binary)
        cv2.waitKey(1)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        # Bug Fix 2: Add constant delay to ensure consistent timing between state transitions
        rospy.sleep(0.1)  # 100ms delay for consistent state transitions
        
        # Bug Fix 1: Use subscriber data instead of polling
        # Wait briefly for subscriber to receive a fresh image
        timeout_counter = 0
        while self.image_data is None and timeout_counter < 50:
            rospy.sleep(0.01)  # Wait 10ms
            timeout_counter += 1
        
        if self.image_data is None:
            print("Warning: No image data received from camera")
            # Return a default state if no image available
            return [0] * 20, False, -100, {}
        
        # Use the most recent image from subscriber
        data = self.image_data

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards based on CURRENT line position (first 10 bins) and FUTURE line position (last 10 bins)
        if not done:
            # Find which bin has the CURRENT line (where state[i] == 1 for i in 0-9)
            current_position = -1
            for i in range(10):
                if state[i] == 1:
                    current_position = i
                    break
            
            # Find which bin has the FUTURE line (where state[i] == 1 for i in 10-19)
            future_position = -1
            for i in range(10, 20):
                if state[i] == 1:
                    future_position = i - 10  # Convert to 0-9 range
                    break
            
            # Initialize reward
            reward = 0
            
            # CURRENT POSITION REWARDS (heavily weighted for center)
            if current_position == -1:
                # No current line detected - HEAVY penalty
                reward -= 50
            elif current_position in [4, 5]:
                # Current line is in the CENTER (bins 4 or 5) - VERY HIGH reward
                reward += 100
                # MASSIVE bonus if moving forward while perfectly centered
                if action == 0:
                    reward += 50
            elif current_position in [3, 6]:
                # Current line is slightly off center - small reward
                reward += 10
            elif current_position in [2, 7]:
                # Current line is more off center - tiny reward
                reward += 2
            else:
                # Current line is at the edges (bins 0, 1, 8, 9) - HEAVY penalty
                reward -= 30
            
            # FUTURE POSITION REWARDS (predictive adjustment)
            if future_position >= 0:
                # Reward for future line also being centered
                if future_position in [4, 5]:
                    reward += 20  # Bonus for upcoming centered line
                elif future_position in [3, 6]:
                    reward += 5
                # Penalty if future line is at edges (encourages preemptive correction)
                elif future_position in [0, 1, 8, 9]:
                    reward -= 10
                
                # Reward for taking corrective action based on future position
                if action == 1 and future_position < 4:  # Turn left if line is drifting left
                    reward += 15
                elif action == 2 and future_position > 5:  # Turn right if line is drifting right
                    reward += 15
        else:
            # Episode terminated (lost the line for too long) - MASSIVE penalty
            reward = -500

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Bug Fix 1: Use subscriber data instead of polling
        # Reset the image data and wait for a fresh image
        self.image_data = None
        timeout_counter = 0
        while self.image_data is None and timeout_counter < 100:
            rospy.sleep(0.01)  # Wait 10ms
            timeout_counter += 1
        
        if self.image_data is None:
            print("Warning: No image data received during reset")
            # Return a default state if no image available
            return [0] * 20
        
        data = self.image_data

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
