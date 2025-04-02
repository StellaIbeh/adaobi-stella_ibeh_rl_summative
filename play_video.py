import os
import sys
import cv2
import glob
import argparse

def list_videos():
    """List all available MP4 videos in the visualization output directory"""
    video_dir = os.path.join(os.getcwd(), 'visualization_output')
    videos = glob.glob(os.path.join(video_dir, '*.mp4'))
    
    if not videos:
        print("No videos found in visualization_output directory")
        return None
    
    print("\nAvailable videos:")
    for i, video in enumerate(videos):
        print(f"{i+1}. {os.path.basename(video)}")
    
    return videos

def play_video(video_path):
    """Play a video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    print(f"\nPlaying: {os.path.basename(video_path)}")
    print("Press 'q' to quit, 'p' to pause/play, 's' to slow down, 'f' to speed up")
    
    frame_delay = 25  # milliseconds between frames (default speed)
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for better viewing in smaller windows
            height, width = frame.shape[:2]
            max_height = 800
            if height > max_height:
                scale = max_height / height
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
            # Display frame
            cv2.imshow('Rehabilitation Assistant Simulation', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Video " + ("paused" if paused else "playing"))
        elif key == ord('s'):
            frame_delay = min(100, frame_delay + 10)
            print(f"Slowing down: {frame_delay}ms delay")
        elif key == ord('f'):
            frame_delay = max(5, frame_delay - 10)
            print(f"Speeding up: {frame_delay}ms delay")
    
    cap.release()
    cv2.destroyAllWindows()

def list_gifs():
    """List all available GIF files in the visualization output directory"""
    video_dir = os.path.join(os.getcwd(), 'visualization_output')
    gifs = glob.glob(os.path.join(video_dir, '*.gif'))
    
    if not gifs:
        print("No GIFs found in visualization_output directory")
        return None
    
    print("\nAvailable GIFs:")
    for i, gif in enumerate(gifs):
        print(f"{i+1}. {os.path.basename(gif)}")
    
    return gifs

def play_gif(gif_path):
    """Display a GIF using imageio and OpenCV"""
    try:
        import imageio
        gif = imageio.mimread(gif_path)
        
        print(f"\nPlaying GIF: {os.path.basename(gif_path)}")
        print("Press 'q' to quit, 'p' to pause/play, 's' to slow down, 'f' to speed up")
        
        # Convert from RGB to BGR for OpenCV
        frames = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        
        frame_delay = 200  # milliseconds between frames (slower for GIFs)
        paused = False
        current_frame = 0
        
        while True:
            if not paused:
                # Get the current frame
                frame = frames[current_frame]
                
                # Resize for better viewing in smaller windows
                height, width = frame.shape[:2]
                max_height = 800
                if height > max_height:
                    scale = max_height / height
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
                # Display frame
                cv2.imshow('Rehabilitation Assistant Simulation (GIF)', frame)
                
                # Move to next frame, loop if at end
                current_frame = (current_frame + 1) % len(frames)
            
            # Handle keyboard input
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("GIF " + ("paused" if paused else "playing"))
            elif key == ord('s'):
                frame_delay = min(500, frame_delay + 50)
                print(f"Slowing down: {frame_delay}ms delay")
            elif key == ord('f'):
                frame_delay = max(50, frame_delay - 50)
                print(f"Speeding up: {frame_delay}ms delay")
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error playing GIF: {e}")

def main():
    parser = argparse.ArgumentParser(description='Play rehabilitation simulation videos and GIFs')
    parser.add_argument('--video', type=str, help='Path to specific video file to play')
    parser.add_argument('--gif', type=str, help='Path to specific GIF file to play')
    parser.add_argument('--format', type=str, choices=['video', 'gif'], 
                        help='Choose whether to list videos or GIFs')
    args = parser.parse_args()
    
    if args.video:
        # Play specific video if provided
        play_video(args.video)
    elif args.gif:
        # Play specific GIF if provided
        play_gif(args.gif)
    else:
        # Determine which format to list based on args or ask user
        format_choice = args.format
        if not format_choice:
            format_choice = input("Which format do you want to view? (video/gif): ").lower()
        
        if format_choice == 'gif':
            # List and play GIFs
            gifs = list_gifs()
            if not gifs:
                return
            
            while True:
                choice = input("\nEnter number to play (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(gifs):
                        play_gif(gifs[idx])
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a number")
        else:
            # Default to listing and playing videos
            videos = list_videos()
            if not videos:
                return
            
            while True:
                choice = input("\nEnter number to play (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(videos):
                        play_video(videos[idx])
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a number")

if __name__ == "__main__":
    main()