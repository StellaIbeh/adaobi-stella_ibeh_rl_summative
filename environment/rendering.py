import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import threading
import imageio  # You'll need to install this: pip install imageio
from datetime import datetime
import cv2  # For video creation fallback

class Renderer:
    def __init__(self):
        self.exercise_progress = 0.0
        self.posture_correctness = True
        self.joint_angles = np.zeros(3)
        self.muscle_activation = np.zeros(3)
        self.running = False
        self.fig = None
        self.display_thread = None
        self.frame_count = 0
        self.output_dir = os.path.join(os.getcwd(), 'visualization_output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frames = []  # To store frames for video creation
        
        # Initialize video writer
        self.video_writer = None
        self.video_path = os.path.join(self.output_dir, f"simulation_{self.session_id}.mp4")
        self.temp_frame_dir = os.path.join(self.output_dir, 'temp_frames')
        os.makedirs(self.temp_frame_dir, exist_ok=True)

    def update_progress(self, progress, correctness, state=None):
        self.exercise_progress = progress
        self.posture_correctness = correctness
        if state is not None:
            self.joint_angles = state[:3]
            self.muscle_activation = state[3:]
        
        # Create the plot if not already created
        if self.fig is None:
            self.setup_plot()
        
        # Update and save the current frame
        self.update_plot()
        self.save_frame()

    def setup_plot(self):
        self.fig = plt.figure(figsize=(12, 8), facecolor='#f0f0f0')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Create layout with subplots
        self.ax_human = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
        self.ax_progress = plt.subplot2grid((3, 3), (0, 2))
        self.ax_angles = plt.subplot2grid((3, 3), (1, 2))
        self.ax_activation = plt.subplot2grid((3, 3), (2, 2))
        
        # Set up figure title
        self.fig.suptitle("Rehabilitation Assistant Visualization", fontsize=16, fontweight='bold')
        
        # Set up axes
        self.ax_human.set_xlim(-1.5, 1.5)
        self.ax_human.set_ylim(-1.5, 1.5)
        self.ax_human.set_aspect('equal')
        self.ax_human.axis('off')
        
        self.ax_progress.set_title("Exercise Progress")
        self.ax_progress.set_xlim(0, 1)
        self.ax_progress.set_ylim(0, 1)
        self.ax_progress.axis('off')
        
        self.ax_angles.set_title("Joint Angles")
        self.ax_angles.set_xlim(0, 3)
        self.ax_angles.set_ylim(-np.pi, np.pi)
        self.ax_angles.set_xticks([0.5, 1.5, 2.5])
        self.ax_angles.set_xticklabels(['Joint 1', 'Joint 2', 'Joint 3'])
        
        self.ax_activation.set_title("Muscle Activation")
        self.ax_activation.set_xlim(0, 3)
        self.ax_activation.set_ylim(0, 1)
        self.ax_activation.set_xticks([0.5, 1.5, 2.5])
        self.ax_activation.set_xticklabels(['Muscle 1', 'Muscle 2', 'Muscle 3'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Initialize plots
        self.progress_bar = self.ax_progress.barh(0.5, 0, height=0.3, color='green')
        self.progress_text = self.ax_progress.text(0.5, 0.5, "0%", 
                                                 horizontalalignment='center',
                                                 verticalalignment='center',
                                                 fontsize=12)
        
        self.angle_bars = self.ax_angles.bar([0.5, 1.5, 2.5], self.joint_angles, width=0.6)
        self.activation_bars = self.ax_activation.bar([0.5, 1.5, 2.5], self.muscle_activation, width=0.6)
        
        # Status indicator for posture correctness
        self.status_text = self.ax_human.text(0, 1.3, "POSTURE: CORRECT", 
                                             horizontalalignment='center',
                                             color='green', fontsize=14,
                                             fontweight='bold')
    
    def update_plot(self):
        # Update progress bar
        self.progress_bar[0].set_width(self.exercise_progress)
        self.progress_text.set_text(f"{self.exercise_progress*100:.1f}%")
        self.progress_text.set_position((self.exercise_progress/2, 0.5))
        
        # Update joint angles
        for i, bar in enumerate(self.angle_bars):
            # Color code bars based on how close they are to optimal position
            # (In rehabilitation, closer to 0 is usually better)
            deviation = abs(self.joint_angles[i])
            normalized_deviation = min(deviation / np.pi, 1.0)  # Scale to [0,1]
            # Color gradient from green (good) to red (bad)
            color = (normalized_deviation, 1.0 - normalized_deviation, 0.0)
            bar.set_height(self.joint_angles[i])
            bar.set_color(color)
        
        # Update muscle activation
        for i, bar in enumerate(self.activation_bars):
            # Color based on activation level - from blue (low) to red (high)
            activation = self.muscle_activation[i]
            bar.set_height(activation)
            # Color: blue for low, purple for medium, red for high activation
            if activation < 0.3:
                color = 'blue'
            elif activation < 0.7:
                color = 'purple'
            else:
                color = 'red'
            bar.set_color(color)
        
        # Update posture correctness
        if self.posture_correctness:
            self.status_text.set_text("POSTURE: CORRECT")
            self.status_text.set_color('green')
        else:
            self.status_text.set_text("POSTURE: INCORRECT")
            self.status_text.set_color('red')
        
        # Draw stick figure based on joint angles
        self.ax_human.clear()
        self.ax_human.set_xlim(-1.5, 1.5)
        self.ax_human.set_ylim(-1.5, 1.5)
        self.ax_human.axis('off')
        
        # Add posture status text back after clearing
        self.status_text = self.ax_human.text(0, 1.3, 
                                             "POSTURE: CORRECT" if self.posture_correctness else "POSTURE: INCORRECT", 
                                             horizontalalignment='center',
                                             color='green' if self.posture_correctness else 'red', 
                                             fontsize=14, fontweight='bold')
        
        # Add current action display
        if hasattr(self, 'current_action'):
            action_text = "ACTION: Unknown"
            action_color = 'gray'
            
            if self.current_action == 0:
                action_text = "ACTION: Adjust Difficulty"
                action_color = 'blue'
            elif self.current_action == 1:
                action_text = "ACTION: Encourage"
                action_color = 'green'
            elif self.current_action == 2:
                action_text = "ACTION: Suggest Break"
                action_color = 'orange'
                
            self.ax_human.text(0, -1.3, action_text,
                              horizontalalignment='center',
                              color=action_color, fontsize=12,
                              fontweight='bold')
        
        # Draw stick figure with color based on posture correctness
        figure_color = 'green' if self.posture_correctness else 'red'
        
        # Head
        head = plt.Circle((0, 0.7), 0.2, fill=False, color=figure_color, linewidth=2)
        self.ax_human.add_patch(head)
        
        # Torso
        self.ax_human.plot([0, 0], [0.5, -0.2], color=figure_color, linewidth=3)
        
        # Arms - use joint angles to determine position
        left_angle = self.joint_angles[0]
        right_angle = self.joint_angles[1]
        
        # Left arm
        left_arm_x = -0.6 * np.cos(left_angle + np.pi/2)
        left_arm_y = 0.3 + 0.6 * np.sin(left_angle + np.pi/2)
        self.ax_human.plot([0, left_arm_x], [0.3, left_arm_y], color=figure_color, linewidth=3)
        
        # Right arm
        right_arm_x = 0.6 * np.cos(right_angle + np.pi/2)
        right_arm_y = 0.3 + 0.6 * np.sin(right_angle + np.pi/2)
        self.ax_human.plot([0, right_arm_x], [0.3, right_arm_y], color=figure_color, linewidth=3)
        
        # Legs
        leg_angle = self.joint_angles[2]
        
        # Left leg
        self.ax_human.plot([0, -0.4], [-0.2, -0.8], color=figure_color, linewidth=3)
        
        # Right leg - use leg angle
        right_leg_x = 0.4 * np.cos(leg_angle)
        right_leg_y = -0.2 - 0.6 * np.sin(leg_angle)
        self.ax_human.plot([0, right_leg_x], [-0.2, right_leg_y], color=figure_color, linewidth=3)
        
        # Add reward information
        if hasattr(self, 'current_reward'):
            reward_text = f"REWARD: {self.current_reward:.1f}"
            reward_color = 'green' if self.current_reward > 0 else 'red'
            self.ax_human.text(0, -1.0, reward_text, 
                             horizontalalignment='center',
                             color=reward_color, fontsize=12, 
                             fontweight='bold')

    def save_frame(self):
        """Save current frame and add to video"""
        self.frame_count += 1
        
        # Save temporary frame for video processing
        temp_filename = os.path.join(self.temp_frame_dir, f"frame_{self.frame_count:04d}.png")
        self.fig.savefig(temp_filename)
        
        # Store the filename for video creation
        self.frames.append(temp_filename)
        
        # Print progress less frequently to avoid console spam
        if self.frame_count % 5 == 0:
            print(f"Captured frame {self.frame_count} for video")

    def finalize_video(self, model_name, posture_type):
        """Create both video and GIF from captured frames with annotations"""
        if len(self.frames) == 0:
            print("No frames to create video from!")
            return None
        
        # Create video and GIF paths for this simulation
        video_path = os.path.join(self.output_dir, f"{model_name}_{posture_type}_{self.session_id}.mp4")
        gif_path = os.path.join(self.output_dir, f"{model_name}_{posture_type}_{self.session_id}.gif")
        
        try:
            # Read the first image to get dimensions
            img = cv2.imread(self.frames[0])
            height, width, layers = img.shape
            size = (width, height)
            
            # 1. Create MP4 Video
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 5.0, size)  # 5 FPS for better viewing
            
            # Store frames for GIF creation
            gif_frames = []
            
            # Write frames to video with annotations
            for i, frame_path in enumerate(self.frames):
                # Read frame
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Add frame counter
                    cv2.putText(frame, f"Frame: {i+1}/{len(self.frames)}", 
                              (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 0), 2)
                    
                    # Add model and posture info
                    cv2.putText(frame, f"Model: {model_name}, Posture: {posture_type}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 0), 2)
                    
                    # Write to video
                    video.write(frame)
                    
                    # Convert BGR to RGB for GIF (imageio expects RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gif_frames.append(rgb_frame)
            
            # Release video writer
            video.release()
            print(f"\nCreated video: {video_path}")
            
            # 2. Create GIF using imageio
            # For better file size, we'll resize frames and use a lower framerate for GIFs
            resized_gif_frames = []
            gif_size = (width // 2, height // 2)  # Half size for GIFs to reduce file size
            
            for frame in gif_frames:
                resized = cv2.resize(frame, gif_size)
                resized_gif_frames.append(resized)
            
            # Save as GIF
            imageio.mimsave(gif_path, resized_gif_frames, fps=3)  # Lower FPS for GIFs
            print(f"Created GIF: {gif_path}")
            
            # Clean up temporary frame files if requested
            for frame_path in self.frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            return video_path
                    
        except Exception as e:
            print(f"Error creating video/GIF: {e}")
            return None

    def create_video(self, output_name="complete_simulation"):
        """Create a master video and GIF from all simulations"""
        all_mp4_files = [f for f in os.listdir(self.output_dir) if f.endswith('.mp4') and f != f"{output_name}.mp4"]
        
        if not all_mp4_files:
            print("No MP4 files found to combine!")
            return None
        
        # Paths for the combined outputs
        combined_video_path = os.path.join(self.output_dir, f"{output_name}.mp4")
        combined_gif_path = os.path.join(self.output_dir, f"{output_name}.gif")
        
        try:
            # 1. Create combined video using FFmpeg
            # Create a temporary file listing all videos to concatenate
            with open('videos_to_concat.txt', 'w') as f:
                for video_file in all_mp4_files:
                    f.write(f"file '{os.path.join(self.output_dir, video_file)}'\n")
            
            # Use FFmpeg to concatenate videos (requires FFmpeg installed)
            os.system(f'ffmpeg -f concat -safe 0 -i videos_to_concat.txt -c copy {combined_video_path}')
            
            # Clean up the temporary file
            if os.path.exists('videos_to_concat.txt'):
                os.remove('videos_to_concat.txt')
            
            print(f"\nCreated combined video: {combined_video_path}")
            
            # 2. Create combined GIF (take first few frames from each video to keep size reasonable)
            try:
                # For the combined GIF, we'll extract key frames from each video
                combined_frames = []
                
                for video_file in all_mp4_files:
                    cap = cv2.VideoCapture(os.path.join(self.output_dir, video_file))
                    
                    # Get basic info about the video
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    sample_interval = max(1, total_frames // 10)  # Take ~10 frames from each video
                    
                    # Extract frames
                    count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if count % sample_interval == 0:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            combined_frames.append(rgb_frame)
                        count += 1
                    
                    cap.release()
                
                # Resize frames for GIF
                gif_size = (combined_frames[0].shape[1] // 2, combined_frames[0].shape[0] // 2)
                resized_combined_frames = [cv2.resize(frame, gif_size) for frame in combined_frames]
                
                # Save combined GIF
                imageio.mimsave(combined_gif_path, resized_combined_frames, fps=3)
                print(f"Created combined GIF: {combined_gif_path}")
                
            except Exception as e:
                print(f"Error creating combined GIF: {e}")
            
            return combined_video_path
            
        except Exception as e:
            print(f"Error combining videos: {e}")
            
            # Fallback to just copying the longest video as the complete simulation
            try:
                largest_video = max(all_mp4_files, key=lambda x: os.path.getsize(os.path.join(self.output_dir, x)))
                import shutil
                shutil.copy(os.path.join(self.output_dir, largest_video), combined_video_path)
                print(f"Fallback: copied {largest_video} as {output_name}.mp4")
                return combined_video_path
            except:
                print("Fallback method also failed.")
                return None

    def set_action_and_reward(self, action, reward):
        """Set the current action and reward for display"""
        self.current_action = action
        self.current_reward = reward

    def run(self):
        # This method is now used just to clean up when done
        if self.fig is not None:
            plt.close(self.fig)