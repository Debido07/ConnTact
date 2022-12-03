# Copyright 2021 Southwest Research Institute
# Licensed under the Apache License, Version 2.0

from conntact.conntask import ConnTask, ConnStep
from conntact.conntact_interface import ConntactInterface
from conntact.conntext import Conntext
from colorama import Back, Fore, Style
import numpy as np
from transitions import Machine
import conntact.assembly_utils as utils


START_STATE    = 'state_start'
APPROACH_STATE = 'state_finding_surface'
FIND_HOLE_STATE = 'state_finding_hole'
INSERTING_PEG_STATE = 'state_inserting_along_axis'
THREAD_STATE = 'thread_state'
COMPLETION_STATE = 'state_completed_insertion'
EXIT_STATE = 'state_exit'
SAFETY_RETRACT_STATE = 'state_safety_retraction'
CONTACT_CAM_PEGS_STATE = 'state_first_insertion'
ROTATE_PEG_STATE = 'state_rotate_peg'

# Trigger names
APPROACH_SURFACE_TRIGGER = 'start approach'
FIND_HOLE_TRIGGER = 'surface found'
INSERT_PEG_TRIGGER = 'hole found'
ASSEMBLY_COMPLETED_TRIGGER = 'assembly completed'
STEP_COMPLETE_TRIGGER = 'next step'
SAFETY_RETRACTION_TRIGGER = 'retract to safety'
RESTART_TEST_TRIGGER = 'restart test'
RUN_LOOP_TRIGGER = 'run looped code'

class BNCTask(ConnTask):

    def __init__(self, conntext, interface, target_frame_name, connfig_name):

        #Declare the official states list here. These will be passed into the machine.
        states = [
            START_STATE,
            APPROACH_STATE,
            FIND_HOLE_STATE, 
            CONTACT_CAM_PEGS_STATE, 
            ROTATE_PEG_STATE,
            INSERTING_PEG_STATE,
            THREAD_STATE, 
            COMPLETION_STATE,
            EXIT_STATE,
            SAFETY_RETRACT_STATE
        ]

        # Define the valid transitions from/to each state. Here's where you define the topology of the state machine.
        # The Machine executes the first transition in this list which matches BOTH the trigger AND the CURRENT state.
        # If no other trigger is set at "self.next_trigger", Conntact will automatically fill in "RUN_LOOP_TRIGGER"
        # which runs the Execute method of the current Step object.
        transitions = [
            {'trigger':APPROACH_SURFACE_TRIGGER  , 'source':START_STATE         , 'dest':APPROACH_STATE         },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':APPROACH_STATE      , 'dest':FIND_HOLE_STATE        },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':FIND_HOLE_STATE     , 'dest':CONTACT_CAM_PEGS_STATE    },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':CONTACT_CAM_PEGS_STATE , 'dest':ROTATE_PEG_STATE      },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':ROTATE_PEG_STATE, 'dest':INSERTING_PEG_STATE      },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':INSERTING_PEG_STATE , 'dest':THREAD_STATE       },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':THREAD_STATE , 'dest':EXIT_STATE       },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':COMPLETION_STATE    , 'dest':EXIT_STATE             },
            {'trigger':SAFETY_RETRACTION_TRIGGER , 'source':'*'                 , 'dest':SAFETY_RETRACT_STATE,
              'unless':'is_already_retracting' },
            {'trigger':STEP_COMPLETE_TRIGGER     , 'source':SAFETY_RETRACT_STATE, 'dest':APPROACH_STATE         },
            {'trigger':RUN_LOOP_TRIGGER          , 'source':'*'                 , 'dest':None, 'after': 'run_step_actions'}
        ]

        self.step_list:dict = { APPROACH_STATE:       (FindSurface, []),
                                FIND_HOLE_STATE:      (SpiralToFindHole, []),
                                CONTACT_CAM_PEGS_STATE:  (FindSurfaceFullCompliant, []),
                                ROTATE_PEG_STATE:     (RotatePeg, [[0,0,359], .20]),
                                INSERTING_PEG_STATE:  (FindSurfaceFullCompliant, []),
                                THREAD_STATE:         (RotatePeg, [[0,0,-359], .35]),
                                SAFETY_RETRACT_STATE: (SafetyRetraction, []),
                                COMPLETION_STATE:     (ExitStep, [])
                                }
        # #Initialize the state machine "Machine" init in your Conntask instance
        ConnTask.__init__(self, conntext, states, transitions, "target_hole_position", connfig_name=connfig_name)

        # set up the precision_peg parameters and read the connfig
        self.readYAML()
        self.tcp_selected = 'tip'
        self.reset_height = self.connfig['task']['restart_height'] / 100

    def read_peg_hole_dimensions(self):
        """Read peg and hole data from YAML configuration file.
        """
        self.hole_depth = self.connfig['objects']['dimensions']['min_insertion_depth'] / 1000
        self.safe_clearance = self.connfig['objects']['dimensions']['safe_clearance'] / 2000;  # = .2 *radial* clearance i.e. on each side.

    def readYAML(self):
        """Read data from job config YAML and make certain calculations for later use. Stores peg frames in dictionary tool_data
        """

        activeTCP = self.connfig['task']['starting_tcp']
        self.read_board_positions()
        self.read_peg_hole_dimensions()

        # Spiral parameters
        self._spiral_params = self.connfig['task']['spiral_params']

        # Calculate transform from TCP to peg corner
        peg_locations = self.connfig['objects']['grasping_locations']

        # Set up tool_data.
        self.conntext.toolData.name = activeTCP
        self.conntext.toolData.validToolsDict = peg_locations

        self.conntext.select_tool(activeTCP)
        print("Select Tool successful!")

        self.surface_height = 0.0 # Starting height assumption
        self.restart_height = self.connfig['task']['restart_height']  # Height to restart

    def read_board_positions(self):
        """ Calculates pose of target hole relative to robot base frame.
        """
        self.target_hole_pose = self.interface.get_transform("target_hole_position", "base_link")
        self.x_pos_offset = self.target_hole_pose.transform.translation.x
        self.y_pos_offset = self.target_hole_pose.transform.translation.y

    def all_states_calc(self):
        self.publish_plotted_values(('state', self.state))
        super().all_states_calc()

    def publish_plotted_values(self, stateInfo) -> None:
        """Publishes critical data for plotting node to process.
        """

        items = dict()
        # Send a dictionary as plain text to expose some additional info
        items["status_dict"] = dict(
            {stateInfo, ('tcp_name', str(self.conntext.toolData.frame_name))})
        if (self.surface_height != 0.0):
            # If we have located the work surface
            items["status_dict"]['surface_height'] = str(self.surface_height)
        items["_average_wrench_world"] = self.conntext._average_wrench_world
        items["average_speed"] = self.conntext.average_speed
        items["current_pose"] = self.conntext.current_pose.transform.translation

        self.interface.publish_plotting_values(items)

    def main(self):
        self.next_trigger, self.switch_state = APPROACH_SURFACE_TRIGGER, True
        self.interface.send_info(Fore.BLACK + Back.GREEN + "Beginning search algorithm. "+Style.RESET_ALL)
        self.algorithm_execute()
        self.interface.send_info("Peg Insertion all done!")

class FindSurface(ConnStep):

    def __init__(self, connTask: ConnTask) -> None:
        ConnStep.__init__(self, connTask)
        self.create_move_policy(move_mode="line",
                                  vector = [0, 0, 1],  #wrt task space
                                  force = [0, 0, -7],
                                  origin = [0,0,0], 
                                  orientation = [0,0,0])


    def exit_conditions(self) -> bool:
        return self.is_static() and self.in_collision()

    def on_exit(self):
        """Executed once, when the change-state trigger is registered.
        """
        # Measure flat surface height and report it to AssemblyBlocks:
        self.task.surface_height = self.conntext.current_pose.transform.translation.z
        return super().on_exit()

class FindSurfaceFullCompliant(ConnStep):
    def __init__(self, connTask: (ConnTask)) -> None:
        ConnStep.__init__(self, connTask)
        self.exitPeriod = 1

        self.create_move_policy(move_mode="free",
                            force = [0, 0, -15])

    def exit_conditions(self) -> bool:
        # self.move_policy.orientation = utils.qToEu(self.conntext.current_pose.transform.rotation)
        self.conntext.interface.send_info("Current force is {}".format(self.conntext.current_wrench.wrench.force), 1)
        return self.is_static() and self.in_collision()

class RotatePeg(ConnStep):
    def __init__(self, connTask: (ConnTask), setpoint, torque_threshhold) -> None:
        ConnStep.__init__(self, connTask)
        self.exitPeriod = .05
        self.create_move_policy(move_mode="free",
                            force = [0, 0, -5])
        self.cycle_period = 14
        self.surface_height = self.conntext.current_pose.transform.translation.z
        self.start_time = self.conntext.interface.get_unified_time(float=True)
        self.setpoint = setpoint
        self.torque_thresh = torque_threshhold

    def exit_conditions(self) -> bool:
        self.conntext.interface.send_info("Current wrench is {}".format(self.conntext.current_wrench.wrench.torque), 1)
        return self.conntext.current_pose.transform.translation.z <= self.surface_height - .002 or \
            abs(self.conntext.current_wrench.wrench.torque.z) > self.torque_thresh

    def execute(self):
        curr_time = self.conntext.interface.get_unified_time(float=True) - self.start_time
        self.move_policy.orientation = self.setpoint


class SpiralToFindHole(ConnStep):
    def __init__(self, connTask: (ConnTask)) -> None:
        ConnStep.__init__(self, connTask)
        self.create_move_policy(move_mode="line",
                    vector = [0, 0, 1],
                    force = [0, 0, -7],
                    origin = [0,0,0])
        self.spiral_params = self.task.connfig['task']['spiral_params']
        self.safe_clearance = self.task.connfig['objects']['dimensions']['safe_clearance']/100 #convert to m
        self.start_time = self.conntext.interface.get_unified_time()
        self.exitPeriod = .025 

    def execute(self):
        '''
        Executed once per loop while this State is active. By default, just runs update_commands to keep the compliance
        motion profile running.
        '''
        self.move_policy.origin = self.get_spiral_search_pose()[0]
        super().execute()

    def exit_conditions(self) -> bool:
        return self.conntext.current_pose.transform.translation.z <= self.task.surface_height - .001

    def get_spiral_search_pose(self):
        """Generates position, orientation offset vectors which describe a plane spiral about z;
        Adds this offset to the current approach vector to create a searching pattern. Constants come from Init;
        x,y vector currently comes from x_ and y_pos_offset variables.
        """
        # frequency=.15, min_amplitude=.002, max_cycles=62.83185
        curr_time = self.conntext.interface.get_unified_time() - self.start_time
        curr_time_numpy = np.double(curr_time.to_sec())
        frequency = self.spiral_params['frequency'] #because we refer to it a lot
        curr_amp = self.spiral_params['min_amplitude'] + self.safe_clearance * \
                   np.mod(2.0 * np.pi * frequency * curr_time_numpy, self.spiral_params['max_cycles']);
        x_pos = curr_amp * np.cos(2.0 * np.pi * frequency * curr_time_numpy)
        y_pos = curr_amp * np.sin(2.0 * np.pi * frequency * curr_time_numpy)
        # These values can remain zero because the pos_vec command is interpreted relative to the task frame:

        # x_pos = x_pos + self.task.x_pos_offset
        # y_pos = y_pos + self.task.y_pos_offset
        z_pos = self.conntext.current_pose.transform.translation.z
        pose_position = [x_pos, y_pos, z_pos]
        # pose_orientation = [0, 1, 0, 0]  #w, x, y, z equiv to XYZ = 180,0,0
        pose_orientation = [1, 0, 0, 0]  #w, x, y, z equiv to XYZ = 0,0,0

        return [pose_position, pose_orientation]

class SafetyRetraction(ConnStep):
    def __init__(self, connTask: (ConnTask)) -> None:
        ConnStep.__init__(self, connTask)
        self.create_move_policy(move_mode="free",
            vector = [0, 0, 1],
            force = [0, 0, 10],
            origin = [0,0,0])

    def exit_conditions(self) -> bool:
        return self.no_force() and self.above_restart_height()

    def above_restart_height(self):
        return self.conntext.current_pose.transform.translation.z > \
               self.task.surface_height + self.task.reset_height

class ExitStep(ConnStep):
    def __init__(self, connTask: (ConnTask)) -> None:
        ConnStep.__init__(self, connTask)
        self.create_move_policy(move_mode="free",
            vector = [0, 0, 1],
            force = [0, 0, 20],
            origin = [0,0,0])

    def exit_conditions(self) -> bool:
        return self.above_restart_height()

    def above_restart_height(self):
        return self.conntext.current_pose.transform.translation.z > \
               self.task.surface_height + self.task.reset_height