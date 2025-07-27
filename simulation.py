import carla
import random
import time
import sys
import os
import numpy as np
import cv2

egg_path = r"C:\path to CARLA .egg file\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg"
if egg_path not in sys.path:
    sys.path.append(egg_path)

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True  
settings.fixed_delta_seconds = 0.05  
world.apply_settings(settings)

blueprint_library = world.get_blueprint_library()

latest_image = None
latest_lidar = None
display_image = None
display_contour = None

vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)
spawn_point = spawn_points[0] if spawn_points else carla.Transform()
vehicle = None
for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        print(f"[INFO] Ego vehicle spawned at {spawn_point.location}")
        break

if not vehicle:
    print("[ERROR] Failed to spawn ego vehicle. All spawn points may be occupied.")
    sys.exit(1)

collision_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

def on_collision(event):
    print(f"COLLISION DETECTED with {event.other_actor.type_id}!")
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))

collision_sensor.listen(on_collision)

spectator = world.get_spectator()

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50')
lidar_bp.set_attribute('rotation_frequency', '20')  
lidar_bp.set_attribute('channels', '64')  
lidar_bp.set_attribute('points_per_second', '100000')  
lidar_bp.set_attribute('upper_fov', '15')
lidar_bp.set_attribute('lower_fov', '-25')

lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
semantic_bp.set_attribute('image_size_x', '400') 
semantic_bp.set_attribute('image_size_y', '300')
semantic_bp.set_attribute('fov', '90')
semantic_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
semantic_camera = world.spawn_actor(semantic_bp, semantic_transform, attach_to=vehicle)

latest_semantic = None

def spawn_pedestrians(world, num_pedestrians=10):
    blueprint_library = world.get_blueprint_library()
    pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')

    spawn_locations = []
    for _ in range(num_pedestrians):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_locations.append(carla.Transform(loc))

    batch = []
    for i in range(min(num_pedestrians, len(spawn_locations))):
        bp = random.choice(pedestrian_blueprints)
        batch.append(carla.command.SpawnActor(bp, spawn_locations[i]))

    results = client.apply_batch_sync(batch, True)
    walkers = [x.actor_id for x in results if not x.error]
    print(f"[INFO] Spawned {len(walkers)} pedestrians.")

def spawn_traffic_and_pedestrians(world, client, num_vehicles=5, num_pedestrians=10):
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(3.5)

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    traffic_vehicles = []

    for i in range(min(num_vehicles, len(spawn_points))):
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle_bp.set_attribute('role_name', 'autopilot')
        transform = spawn_points[i]

        npc = world.try_spawn_actor(vehicle_bp, transform)
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            traffic_vehicles.append(npc)

    print(f"[INFO] Spawned {len(traffic_vehicles)} moving NPC vehicles.")

    spawn_pedestrians(world, num_pedestrians)

spawn_traffic_and_pedestrians(world, client, num_vehicles=5, num_pedestrians=10)

def calculate_lane_offset(semantic_image):
    if semantic_image is None:
        return 0.0

    img = np.frombuffer(semantic_image.raw_data, dtype=np.uint8)
    img = img.reshape((semantic_image.height, semantic_image.width, 4))
    road_mask = (img[:, :, 2] == 6) 

    height, width = road_mask.shape
    center_y = int(height * 0.75)  

    road_line = road_mask[center_y]
    road_indices = np.where(road_line)[0]

    if len(road_indices) == 0:
        return 0.0  

    road_center = np.mean(road_indices)
    image_center = width / 2

    offset = (image_center - road_center) / (width / 2)
    return offset


def on_semantic(image):
    global latest_semantic
    latest_semantic = image

semantic_camera.listen(on_semantic)


def detect_potholes(image, semantic_image=None):
    global display_image, display_contour
    
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))
    img_array = img_array[:, :, :3]  # RGB
    
    display_image = np.copy(img_array)
    
    road_mask = None
    if semantic_image is not None:
        sem_array = np.frombuffer(semantic_image.raw_data, dtype=np.uint8)
        sem_array = sem_array.reshape((semantic_image.height, semantic_image.width, 4))
        
        road_mask = np.zeros((semantic_image.height, semantic_image.width), dtype=np.uint8)
        road_mask[sem_array[:,:,2] == 7] = 255  
        
        if road_mask.shape != (img_array.shape[0], img_array.shape[1]):
            road_mask = cv2.resize(road_mask, (img_array.shape[1], img_array.shape[0]))
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    if road_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=road_mask)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    display_contour = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    
    pothole_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000 and area < 15000:  
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if 0.5 < aspect_ratio < 2.0 and solidity > 0.7:
                if y > img_array.shape[0] * 0.5:
                    pothole_contours.append(cnt)
    
    cv2.drawContours(display_image, pothole_contours, -1, (0, 0, 255), 3)
    
    center_potholes = []
    for cnt in pothole_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        contour_center_x = x + w//2
        if img_array.shape[1]//3 < contour_center_x < (img_array.shape[1]*2)//3:
            center_potholes.append(cnt)
            cv2.drawContours(display_image, [cnt], -1, (0, 255, 0), 3)
    
    return len(center_potholes) > 0

def detect_obstacles(lidar_data, semantic_image=None):
    
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = np.reshape(points, (int(len(points) / 4), 4))
    
    front_sector = points[(points[:, 0] > 0) & (points[:, 0] < 20.0) & 
                         (points[:, 1] > -5.0) & (points[:, 1] < 5.0) &
                         (points[:, 2] > 0.1) & (points[:, 2] < 3.0)]  
    
    if semantic_image is not None:
        pass
    
    if len(front_sector) < 10:
        return 'no_obstacle'
    
    close_front = front_sector[(front_sector[:, 0] < 8.0)] 
    mid_front = front_sector[(front_sector[:, 0] >= 8.0) & (front_sector[:, 0] < 15.0)]  
    far_front = front_sector[(front_sector[:, 0] >= 15.0)]  
    
    left_close = close_front[close_front[:, 1] < -1.0]
    center_close = close_front[(close_front[:, 1] >= -1.0) & (close_front[:, 1] <= 1.0)]
    right_close = close_front[close_front[:, 1] > 1.0]
    
    left_mid = mid_front[mid_front[:, 1] < -1.5]
    center_mid = mid_front[(mid_front[:, 1] >= -1.5) & (mid_front[:, 1] <= 1.5)]
    right_mid = mid_front[mid_front[:, 1] > 1.5]
    
    if len(center_close) > 30:  # Immediate obstacle straight ahead
        return 'center_immediate'
    elif len(left_close) > 30 and len(right_close) > 40:  # Obstacles on both sides
        return 'narrow_passage'
    elif len(center_mid) > 50:  # Obstacle ahead in mid-range
        return 'center'
    elif len(left_close) > 30:  # Obstacle on left side
        return 'left'
    elif len(right_close) > 30:  # Obstacle on right side
        return 'right'
    elif len(left_mid) > 50:  # Mid-range obstacle on left
        return 'left_ahead'
    elif len(right_mid) > 50:  # Mid-range obstacle on right
        return 'right_ahead'
    
    return 'no_obstacle'
    import time

    last_pothole_time = 0
    pothole_cooldown = 2  

    while True:
        world.tick()

        semantic_img = latest_semantic
        pothole_detected = detect_potholes(latest_image, semantic_img)
        obstacle_zone = detect_obstacles(latest_lidar, semantic_img)
        offset = calculate_lane_offset(semantic_img)

        current_time = time.time()
        control = carla.VehicleControl()

        if pothole_detected:
            print("Pothole detected ahead! Slowing down.")
            control.throttle = 0.2
            control.brake = 0.6
            last_pothole_time = current_time
        elif current_time - last_pothole_time < pothole_cooldown:
            print("Caution: recently braked. Holding brake briefly...")
            control.throttle = 0.2
            control.brake = 0.4
        else:
            control.brake = 0.0
            control.throttle = 0.4
            print("Path is clear. Accelerating...")

        if 'last_steer' not in globals():
            last_steer = 0.0  

            steer_gain = 0.5 
            raw_steer = np.clip(-offset * steer_gain, -1.0, 1.0)
            alpha = 0.1  
            control.steer = alpha * raw_steer + (1 - alpha) * last_steer
            last_steer = control.steer

        if abs(offset) > 0.2:
            print(f"[Lane Recovery] Adjusting back to center. Offset: {offset:.2f}")

        vehicle.apply_control(control)

        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))


def is_on_road(semantic_image):
    if semantic_image is None:
        return True  
    
    sem_array = np.frombuffer(semantic_image.raw_data, dtype=np.uint8)
    sem_array = sem_array.reshape((semantic_image.height, semantic_image.width, 4))
    
    bottom_center = sem_array[int(0.9*semantic_image.height):, int(0.4*semantic_image.width):int(0.6*semantic_image.width)]
    
    road_pixels = np.sum(bottom_center[:,:,2] == 7)
    total_pixels = bottom_center.shape[0] * bottom_center.shape[1]
    
    return road_pixels > (total_pixels * 0.5)

def process_terrain_and_control(image, lidar_data, semantic_image=None):
    if image is None or lidar_data is None:
        return  

    pothole_detected = detect_potholes(image, semantic_image)
    obstacle_direction = detect_obstacles(lidar_data, semantic_image)
    on_road = is_on_road(semantic_image) if semantic_image is not None else True

    control = carla.VehicleControl()
    control.hand_brake = False

    control.throttle = 0.5
    control.brake = 0.0
    control.steer = 0.0

    # Off-Road Recovery
    if not on_road:
        print("OFF-ROAD DETECTED! Steering back to road.")
        control.throttle = 0.1
        control.brake = 0.7
        control.steer = 0.6  
        vehicle.apply_control(control)
        return

    # Immediate Obstacle Ahead
    if obstacle_direction == 'center_immediate':
        print("EMERGENCY STOP! Obstacle very close ahead.")
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        vehicle.apply_control(control)
        return

    # Narrow Passage Ahead
    elif obstacle_direction == 'narrow_passage':
        print("Caution: Navigating narrow passage.")
        control.throttle = 0.15
        control.brake = 0.2
        control.steer = 0.0

    # Obstacle Mid-range in Center
    elif obstacle_direction == 'center':
        print("Obstacle ahead. Checking for side clearance.")
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(len(points) / 4), 4))

        left_clearance = points[(points[:, 0] > 0) & (points[:, 0] < 10.0) & (points[:, 1] < -2.0)]
        right_clearance = points[(points[:, 0] > 0) & (points[:, 0] < 10.0) & (points[:, 1] > 2.0)]

        if len(left_clearance) > len(right_clearance):
            print("Steering left to avoid obstacle.")
            control.throttle = 0.25
            control.brake = 0.2
            control.steer = -0.3
        else:
            print("Steering right to avoid obstacle.")
            control.throttle = 0.25
            control.brake = 0.2
            control.steer = 0.3

    # Side Obstacles
    elif obstacle_direction == 'left_ahead':
        print("Slowing down: Obstacle on left.")
        control.throttle = 0.2
        control.brake = 0.1
        control.steer = 0.3
    elif obstacle_direction == 'right_ahead':
        print("Slowing down: Obstacle on right.")
        control.throttle = 0.2
        control.brake = 0.1
        control.steer = -0.3

    # Pothole Detected
    elif pothole_detected:
        print("Pothole detected ahead! Slowing down.")
        control.throttle = 0.2
        control.brake = 0.3
        control.steer = 0.2

    # NORMAL CRUISING
    else:
        control.throttle = 0.4
        control.brake = 0.0
        control.steer = 0.0

    # Lane Center Correction
    lane_offset = calculate_lane_offset(semantic_image)
    if abs(lane_offset) > 0.1:
        print(f"[Lane Recovery] Adjusting back to center. Offset: {lane_offset:.2f}")
        control.steer = -lane_offset * 0.5  # smooth lane re-centering


    vehicle.apply_control(control)

def spawn_npc_vehicles():
    vehicles = []
    for i in range(10):
        npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
        try:
            if i < len(spawn_points):
                if spawn_points[i].location.distance(spawn_point.location) > 10:  
                    npc = world.try_spawn_actor(npc_bp, spawn_points[i])
                    if npc:
                        npc.set_autopilot(True)
                        vehicles.append(npc)
                        print(f"NPC vehicle spawned at {npc.get_location()}")
        except:
            continue
    return vehicles

def spawn_pedestrians():
    walkers = []
    walker_controllers = []
    
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    
    for i in range(10):
        walker_bp = random.choice(walker_bps)
        
        spawn_attempts = 0
        while spawn_attempts < 10:
            rand_x = random.uniform(-50, 50)
            rand_y = random.uniform(-50, 50)
            walker_spawn = carla.Transform(
                spawn_point.location + carla.Location(x=rand_x, y=rand_y, z=0.5),
                carla.Rotation()
            )
            
            walker = world.try_spawn_actor(walker_bp, walker_spawn)
            if walker:
                walkers.append(walker)
                print(f"Pedestrian spawned at {walker.get_location()}")
                break
            
            spawn_attempts += 1
    
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    for walker in walkers:
        controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
        if controller:
            walker_controllers.append(controller)
    
    for controller in walker_controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
    
    return walkers, walker_controllers

def update_spectator_chase_view():
    vehicle_transform = vehicle.get_transform()
    forward_vector = vehicle_transform.get_forward_vector()
    backward_vector = -1 * forward_vector  
    chase_position = vehicle_transform.location + (backward_vector * 10) 
    chase_position.z += 5 
    
    direction = vehicle_transform.location - chase_position
    rotation = carla.Rotation()
    
    rotation.yaw = np.degrees(np.arctan2(direction.y, direction.x))
    rotation.pitch = -15
    
    spectator_transform = carla.Transform(chase_position, rotation)
    spectator.set_transform(spectator_transform)

def on_camera(image):
    global latest_image
    latest_image = image
    if latest_lidar and latest_semantic:
        process_terrain_and_control(image, latest_lidar, latest_semantic)

def on_lidar(lidar_data):
    global latest_lidar
    latest_lidar = lidar_data
    if latest_image and latest_semantic:
        process_terrain_and_control(latest_image, lidar_data, latest_semantic)

camera.listen(on_camera)
lidar.listen(on_lidar)

# Camera and contour images display
def display_windows():
    global display_image, display_contour
    
    if display_image is not None:
        display_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Camera View with Detections', display_rgb)
    
    if display_contour is not None:
        cv2.imshow('Edge Detection', display_contour)
    
    cv2.waitKey(1)

try:
    print("Setting up the environment...")
    
    npc_vehicles = spawn_npc_vehicles()
    walkers, walker_controllers = spawn_pedestrians()
    
    print("Simulation running... Press Ctrl+C to quit.")
    
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
    
    frame_count = 0
    while True:
        world.tick()
        
        update_spectator_chase_view()

        frame_count += 1
        if frame_count % 20 == 0:
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            print(f"Vehicle Speed: {speed:.2f} km/h | Position: {vehicle.get_location()}")
        
        time.sleep(0.01)

# Destroy all actors
except KeyboardInterrupt:
    print("Cleaning up...")
    cv2.destroyAllWindows()
    
    for controller in walker_controllers:
        controller.stop()
        controller.destroy()
    for walker in walkers:
        walker.destroy()
    
    for npc in npc_vehicles:
        npc.destroy()
    
    camera.destroy()
    lidar.destroy()
    semantic_camera.destroy()
    collision_sensor.destroy()
    vehicle.destroy()
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    
    print("All actors cleaned up.")
