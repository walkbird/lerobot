# To Run on the host
'''
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
'''

# To Run the teleop:
'''
PYTHONPATH=src python -m examples.xlerobot.teleoperate_PS4
'''

import time
import numpy as np
import math
import pygame

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics

# Keymaps (semantic action: controller mapping) - Intuitive human control
LEFT_KEYMAP = {
    # Left stick controls left arm XY (when not pressed)
    'x+': 'left_stick_up', 'x-': 'left_stick_down',
    'y+': 'left_stick_right', 'y-': 'left_stick_left',
    # Left stick pressed controls left arm shoulder_pan
    'shoulder_pan+': 'left_stick_pressed_right', 'shoulder_pan-': 'left_stick_pressed_left',
    # L1 pressed controls left arm pitch and wrist_roll
    'pitch+': 'l1_up', 'pitch-': 'l1_down',
    'wrist_roll+': 'l1_right', 'wrist_roll-': 'l1_left',
    # L2 controls left gripper
    'gripper+': 'l2',
    # Head motors
    "head_motor_1+": 'circle', "head_motor_1-": 'square',
    "head_motor_2+": 'cross', "head_motor_2-": 'triangle',
}
RIGHT_KEYMAP = {
    # Right stick controls right arm XY (when not pressed)
    'x+': 'right_stick_up', 'x-': 'right_stick_down',
    'y+': 'right_stick_right', 'y-': 'right_stick_left',
    # Right stick pressed controls right arm shoulder_pan
    'shoulder_pan+': 'right_stick_pressed_right', 'shoulder_pan-': 'right_stick_pressed_left',
    # R1 pressed controls right arm pitch and wrist_roll
    'pitch+': 'r1_up', 'pitch-': 'r1_down',
    'wrist_roll+': 'r1_right', 'wrist_roll-': 'r1_left',
    # R2 controls right gripper
    'gripper+': 'r2',
}

# Base control keymap - Only forward/backward and rotate left/right
BASE_KEYMAP = {
    'forward': 'dpad_down', 'backward': 'dpad_up',
    'rotate_left': 'dpad_left', 'rotate_right': 'dpad_right',
}

# Global reset key for all components
RESET_KEY = 'share'

# Enable to print controller debugging information at runtime
DEBUG_CONTROLLER = False

_previous_debug_state = {
    "left_actions": frozenset(),
    "right_actions": frozenset(),
    "buttons": tuple(),
    "hat": (0, 0),
    "axes": tuple(),
    "base": tuple(),
    "speed": None,
}

# Axis/button index maps for PS4 controllers (pygame 2.x reference layout documented in SDL2)
PS4_AXIS_INDICES = {
    "left_x": (0,),
    "left_y": (1,),
    "right_x": (2,),
    "right_y": (3,),
    "l2": (4,),
    "r2": (5,),
}

PS4_BUTTON_INDICES = {
    "cross": (0,),
    "circle": (1,),
    "square": (2,),
    "triangle": (3,),
    "share": (4,),
    "ps": (5,),
    "options": (6,),
    "l3": (7,),
    "r3": (8,),
    "l1": (9,),
    "r1": (10,),
    "dpad_up": (11,),
    "dpad_down": (12,),
    "dpad_left": (13,),
    "dpad_right": (14,),
    "touchpad": (15,),
    "l2_button": (),  # No dedicated digital button in reference layout
    "r2_button": (),
}


LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}


def discover_serial_ports():
    """Return available serial port device paths sorted alphabetically."""
    try:
        from serial.tools import list_ports
    except Exception as exc:
        return [], exc

    ports = {port.device for port in list_ports.comports() if getattr(port, "device", None)}
    return sorted(ports), None

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 1
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        if key_state.get('head_motor_1+'):
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_1-'):
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_2+'):
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if key_state.get('head_motor_2-'):
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=1):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix  # To distinguish left and right arm
        self.kp = kp
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        # Set the degree step and xy step
        self.degree_step = 2
        self.xy_step = 0.005
        # Set target positions to zero for P control
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()  # Use copy to avoid reference issues
        
        # Reset kinematic variables to their initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Don't let handle_keys recalculate wrist_flex - set it explicitly
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        # Joint increments
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        
        # Gripper control with auto-close functionality
        if key_state.get('gripper+'):
            # Trigger pressed - open gripper (0.1)
            self.target_positions["gripper"] = 2
            if DEBUG_CONTROLLER:
                print(f"[{self.prefix}] gripper: CLOSED")
        else:
            self.target_positions["gripper"] = 90
            if DEBUG_CONTROLLER:
                print(f"[{self.prefix}] gripper: auto-opening to {self.target_positions['gripper']:.1f}")
        
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")

        # XY plane (IK)
        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step
            moved = True
            print(f"[{self.prefix}] x+: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('x-'):
            self.current_x -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x-: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('y+'):
            self.current_y += self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y+: {self.current_y:.4f}")
        if key_state.get('y-'):
            self.current_y -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y-: {self.current_y:.4f}")
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3
            print(f"[{self.prefix}] shoulder_lift: {joint2}, elbow_flex: {joint3}")

        # Wrist flex is always coupled to pitch and the other two
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            -self.target_positions["elbow_flex"]
            + self.pitch
        )
        # print(f"[{self.prefix}] wrist_flex: {self.target_positions['wrist_flex']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action
    

# --- PS4 Controller Mapping ---
def read_ps4_snapshot(joystick):
    """
    Capture current PS4 controller state (axes, buttons, hat).
    """
    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    hat = joystick.get_hat(0) if joystick.get_numhats() > 0 else (0, 0)
    return {"axes": axes, "buttons": buttons, "hat": hat}


def _axis_value(snapshot, axis_key):
    """Return axis value across potential index mappings."""
    axes = snapshot.get("axes", [])
    indices = PS4_AXIS_INDICES.get(axis_key, ())
    if isinstance(indices, int):
        indices = (indices,)
    for idx in indices:
        if idx is None:
            continue
        if idx < len(axes):
            return axes[idx]
    return 0.0


def _button_pressed(snapshot, button_key):
    """Check whether any candidate index for the button is pressed."""
    buttons = snapshot.get("buttons", [])
    indices = PS4_BUTTON_INDICES.get(button_key, ())
    if isinstance(indices, int):
        indices = (indices,)
    for idx in indices:
        if idx is None:
            continue
        if idx < len(buttons) and buttons[idx]:
            return True
    return False


def _dpad_direction_pressed(snapshot, direction):
    """Return True if the requested D-pad direction is active."""
    hat = snapshot.get("hat", (0, 0))
    if direction == "up" and hat[1] == 1:
        return True
    if direction == "down" and hat[1] == -1:
        return True
    if direction == "left" and hat[0] == -1:
        return True
    if direction == "right" and hat[0] == 1:
        return True

    button_key = f"dpad_{direction}"
    return _button_pressed(snapshot, button_key)


def get_ps4_key_state(snapshot, keymap):
    """
    Map PS4 controller state to semantic action booleans using the provided keymap.
    """
    left_x = _axis_value(snapshot, "left_x")
    left_y = _axis_value(snapshot, "left_y")
    right_x = _axis_value(snapshot, "right_x")
    right_y = _axis_value(snapshot, "right_y")
    l2_axis = _axis_value(snapshot, "l2")
    r2_axis = _axis_value(snapshot, "r2")

    # Get stick pressed states and shoulder button states
    left_stick_pressed = _button_pressed(snapshot, "l3")
    right_stick_pressed = _button_pressed(snapshot, "r3")
    l1_pressed = _button_pressed(snapshot, "l1")
    r1_pressed = _button_pressed(snapshot, "r1")
    l2_button = _button_pressed(snapshot, "l2_button")
    r2_button = _button_pressed(snapshot, "r2_button")

    # Map controller state to semantic actions
    state = {}
    for action, control in keymap.items():
        if control == 'l2':
            state[action] = (l2_axis > 0.5) or l2_button
        elif control == 'r2':
            state[action] = (r2_axis > 0.5) or r2_button
        elif control == 'cross':
            state[action] = _button_pressed(snapshot, "cross")
        elif control == 'circle':
            state[action] = _button_pressed(snapshot, "circle")
        elif control == 'square':
            state[action] = _button_pressed(snapshot, "square")
        elif control == 'triangle':
            state[action] = _button_pressed(snapshot, "triangle")
        elif control == 'share':
            state[action] = _button_pressed(snapshot, "share")
        elif control == 'dpad_up':
            state[action] = _dpad_direction_pressed(snapshot, "up")
        elif control == 'dpad_down':
            state[action] = _dpad_direction_pressed(snapshot, "down")
        elif control == 'dpad_left':
            state[action] = _dpad_direction_pressed(snapshot, "left")
        elif control == 'dpad_right':
            state[action] = _dpad_direction_pressed(snapshot, "right")
        # Left stick controls (when not pressed)
        elif control == 'left_stick_up':
            state[action] = (not left_stick_pressed) and (not l1_pressed) and (left_y < -0.5)
        elif control == 'left_stick_down':
            state[action] = (not left_stick_pressed) and (not l1_pressed) and (left_y > 0.5)
        elif control == 'left_stick_left':
            state[action] = (not left_stick_pressed) and (not l1_pressed) and (left_x < -0.5)
        elif control == 'left_stick_right':
            state[action] = (not left_stick_pressed) and (not l1_pressed) and (left_x > 0.5)
        # Right stick controls (when not pressed) - Fixed axis mapping
        elif control == 'right_stick_up':
            state[action] = (not right_stick_pressed) and (not r1_pressed) and (right_y < -0.5)
        elif control == 'right_stick_down':
            state[action] = (not right_stick_pressed) and (not r1_pressed) and (right_y > 0.5)
        elif control == 'right_stick_left':
            state[action] = (not right_stick_pressed) and (not r1_pressed) and (right_x < -0.5)
        elif control == 'right_stick_right':
            state[action] = (not right_stick_pressed) and (not r1_pressed) and (right_x > 0.5)
        # Left stick pressed controls
        elif control == 'left_stick_pressed_right':
            state[action] = left_stick_pressed and (not l1_pressed) and (left_x > 0.5)
        elif control == 'left_stick_pressed_left':
            state[action] = left_stick_pressed and (not l1_pressed) and (left_x < -0.5)
        # Right stick pressed controls - Fixed axis mapping
        elif control == 'right_stick_pressed_right':
            state[action] = right_stick_pressed and (not r1_pressed) and (right_x > 0.5)
        elif control == 'right_stick_pressed_left':
            state[action] = right_stick_pressed and (not r1_pressed) and (right_x < -0.5)
        # L1 pressed controls (only when stick is moved)
        elif control == 'l1_up':
            state[action] = l1_pressed and (left_y < -0.5)
        elif control == 'l1_down':
            state[action] = l1_pressed and (left_y > 0.5)
        elif control == 'l1_right':
            state[action] = l1_pressed and (left_x > 0.5)
        elif control == 'l1_left':
            state[action] = l1_pressed and (left_x < -0.5)
        # R1 pressed controls (only when stick is moved)
        elif control == 'r1_up':
            state[action] = r1_pressed and (right_y < -0.5)
        elif control == 'r1_down':
            state[action] = r1_pressed and (right_y > 0.5)
        elif control == 'r1_right':
            state[action] = r1_pressed and (right_x > 0.5)
        elif control == 'r1_left':
            state[action] = r1_pressed and (right_x < -0.5)
        else:
            state[action] = False
    return state


def debug_print_controller_state(snapshot, left_state, right_state, base_action, speed_multiplier):
    """
    Print controller debug info when state changes to avoid spamming the console.
    """
    if not DEBUG_CONTROLLER:
        return

    global _previous_debug_state

    axes = snapshot.get("axes", [])
    buttons = snapshot.get("buttons", [])
    hat = snapshot.get("hat", (0, 0))

    left_actions = frozenset(action for action, pressed in left_state.items() if pressed)
    if left_actions != _previous_debug_state["left_actions"]:
        left_output = sorted(left_actions) if left_actions else []
        print(f"[DEBUG][LEFT] Active actions: {left_output}")
        _previous_debug_state["left_actions"] = left_actions

    right_actions = frozenset(action for action, pressed in right_state.items() if pressed)
    if right_actions != _previous_debug_state["right_actions"]:
        right_output = sorted(right_actions) if right_actions else []
        print(f"[DEBUG][RIGHT] Active actions: {right_output}")
        _previous_debug_state["right_actions"] = right_actions

    pressed_buttons = tuple(i for i, pressed in enumerate(buttons) if pressed)
    if pressed_buttons != _previous_debug_state["buttons"]:
        print(f"[DEBUG][BUTTONS] Pressed indices: {pressed_buttons if pressed_buttons else ('-',)}")
        _previous_debug_state["buttons"] = pressed_buttons

    if hat != _previous_debug_state["hat"]:
        print(f"[DEBUG][HAT] Position: {hat}")
        _previous_debug_state["hat"] = hat

    active_axes = tuple((idx, round(value, 2)) for idx, value in enumerate(axes) if abs(value) > 0.2)
    if active_axes != _previous_debug_state["axes"]:
        if active_axes:
            axis_str = ", ".join(f"{idx}:{val:+.2f}" for idx, val in active_axes)
        else:
            axis_str = "-"
        print(f"[DEBUG][AXES] Active: {axis_str}")
        _previous_debug_state["axes"] = active_axes

    base_items = tuple(sorted((key, round(value, 3)) for key, value in base_action.items()))
    if base_items != _previous_debug_state["base"]:
        base_output = dict(base_items)
        print(f"[DEBUG][BASE] Action: {base_output}")
        _previous_debug_state["base"] = base_items

    if speed_multiplier != _previous_debug_state["speed"]:
        print(f"[DEBUG][BASE] Speed multiplier: {speed_multiplier}")
        _previous_debug_state["speed"] = speed_multiplier

def get_base_action(snapshot, robot):
    """
    Get base action from PS4 controller input - simplified to only forward/backward and rotate.
    """
    # Get pressed keys for base control
    pressed_keys = set()
    
    # Map controller inputs to keyboard-like keys for base control
    if _dpad_direction_pressed(snapshot, "up"):   # D-pad up
        pressed_keys.add('k')  # Forward
    if _dpad_direction_pressed(snapshot, "down"):  # D-pad down
        pressed_keys.add('i')  # Backward
    if _dpad_direction_pressed(snapshot, "left"):  # D-pad left
        pressed_keys.add('u')  # Rotate left
    if _dpad_direction_pressed(snapshot, "right"):   # D-pad right
        pressed_keys.add('o')  # Rotate right
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action

def get_base_speed_control(snapshot):
    """
    Get base speed control from PS4 controller - L1 for speed decrease, R1 for speed increase.
    Returns speed multiplier (1.0, 2.0, or 3.0) and prints current speed level.
    """
    # Read controller state
    l1_pressed = _button_pressed(snapshot, "l1")
    r1_pressed = _button_pressed(snapshot, "r1")
    
    # Get current speed level from global variable
    global current_base_speed_level
    if 'current_base_speed_level' not in globals():
        current_base_speed_level = 1  # Default speed level
    
    # Speed control logic
    if l1_pressed and not r1_pressed:
        # L1 pressed alone - decrease speed
        if current_base_speed_level > 1:
            current_base_speed_level -= 1
            print(f"[BASE] Speed decreased to level {current_base_speed_level}")
    elif r1_pressed and not l1_pressed:
        # R1 pressed alone - increase speed
        if current_base_speed_level < 3:
            current_base_speed_level += 1
            print(f"[BASE] Speed increased to level {current_base_speed_level}")
    
    # Map speed level to multiplier
    speed_multiplier = float(current_base_speed_level)
    
    return speed_multiplier


def main():
    FPS = 30
    available_ports, ports_error = discover_serial_ports()
    if ports_error:
        print(f"[MAIN] Could not auto-detect serial ports ({ports_error}). Please enter ports manually.")
    elif available_ports:
        print(f"[MAIN] Detected serial ports: {available_ports}")
    else:
        print("[MAIN] No serial ports detected automatically. Connect the robot or run `lerobot-find-port`.")

    defaults = XLerobotConfig()

    def pick_default(original, candidates, index):
        if original in candidates:
            return original
        if index < len(candidates):
            return candidates[index]
        return original

    port1_default = pick_default(defaults.port1, available_ports, 0)
    port2_default = pick_default(defaults.port2, available_ports, 1)

    def prompt_port(label, default_value):
        response = input(f"{label} [{default_value}]: ").strip()
        return response or default_value

    port1 = prompt_port("Enter USB port for the left arm/head bus", port1_default)
    port2 = prompt_port("Enter USB port for the right arm/base bus", port2_default)

    robot = None
    robot_config = None

    while True:
        robot_config = XLerobotConfig(port1=port1, port2=port2)
        robot = XLerobot(robot_config)

        try:
            robot.connect()
            print(f"[MAIN] Successfully connected to robot on ports {port1} and {port2}")
            break
        except ConnectionError as e:
            print(f"[MAIN] Failed to connect with ports {port1} and {port2}: {e}")
            available_ports, ports_error = discover_serial_ports()
            if ports_error:
                print(f"[MAIN] Could not refresh serial port list ({ports_error}).")
            elif available_ports:
                print(f"[MAIN] Updated serial port list: {available_ports}")
            else:
                print("[MAIN] Still no serial ports detected automatically.")

            retry = input("Retry with different ports? (y/n): ").strip().lower()
            if retry not in ("y", "yes"):
                print("Exiting without connecting to the robot.")
                return

            port1 = prompt_port("Enter USB port for the left arm/head bus", port1)
            port2 = prompt_port("Enter USB port for the right arm/base bus", port2)
        except Exception as e:
            print(f"[MAIN] Failed to connect to robot: {e}")
            print(robot_config)
            print(robot)
            return

    init_rerun(session_name="xlerobot_teleop_ps4")

    # Init PS4 controller
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No PS4 controller detected!")
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[MAIN] Using controller: {joystick.get_name()}")

    # Init the arm and head instances
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = SimpleHeadControl(obs)

    # Move both arms and head to zero position at start
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)

    try:
        while True:
            pygame.event.pump()
            snapshot = read_ps4_snapshot(joystick)
            left_key_state = get_ps4_key_state(snapshot, LEFT_KEYMAP)
            right_key_state = get_ps4_key_state(snapshot, RIGHT_KEYMAP)
            
            # Check for global reset (share button)
            global_reset = _button_pressed(snapshot, "share")
            
            # Handle global reset for all components
            if global_reset:
                print("[MAIN] Global reset triggered!")
                left_arm.move_to_zero_position(robot)
                right_arm.move_to_zero_position(robot)
                head_control.move_to_zero_position(robot)
                continue

            # Handle both arms separately and simultaneously
            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)  # Head controlled by left arm keymap

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot)

            # Get base action and speed control from controller
            base_action = get_base_action(snapshot, robot)
            speed_multiplier = get_base_speed_control(snapshot)
            
            # Apply speed multiplier to base actions if they exist
            if base_action:
                for key in base_action:
                    if 'vel' in key or 'velocity' in key:  # Apply to velocity commands
                        base_action[key] *= speed_multiplier

            debug_print_controller_state(snapshot, left_key_state, right_key_state, base_action, speed_multiplier)

            # Merge all actions
            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, action)
    finally:
        robot.disconnect()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()
