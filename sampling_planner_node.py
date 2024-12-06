import rclpy
from rclpy.node import Node
from rclpy.callback_groups import CallbackGroup
from rclpy.clock import ROSClock
from tier4_planning_msgs.msg import Trajectory, TrajectoryPoint

class DiffPlan(Node):

    def __init__(self):
        super().__init__(
            node_name="DiffPlan",
            namespace="/core/planning",
        )
        self.wait_for_state_estimate_group = WaitForStateEstimate(self) # TODO: to be implemented
        self.ros_clock = ROSClock()

        self.raceline = {}
        
        self.request_counter = 0
        self.wait_counter = 0

        # subsribe and publish
        self.performance_trajectory_publisher = self.create_publisher(
            msg_type=Trajectory, topic="/core/planning/target_trajectory/trajectory", qos_profile=1
        )
    
        time_period = 0.01
        self.timer = self.create_timer(
            timer_period_sec=time_period,
            callback=self.timer_callback,
            callback_group=self.wait_for_state_estimate_group,
            clock=self.ros_clock,
        )

    def timer_callback(self):
        # TODO: logging message
        # TODO: check declare_and_update_parameters
        # TODO: request_counter to check the input/planning request

        # Check if receiving raceline input
        if not self.raceline:
            self.wait_counter+=1
            if self.wait_counter % 20 == 0:
                print("Waiting for raceline . . .") # TODO: logging message
            return
        
        # TODO: Local Planner

    def publish_performance_trajectory(self, performance_trajectory: dict):
        # TODO: logging message

        traj_msg = Trajectory()
        self.performance_trajectory_publisher.publish(traj_msg)

class WaitForStateEstimate(CallbackGroup):
    def __init__(self, node) -> None:
        super().__init__()
        self.state_estimate_valid = False
        self.handshake_finished = False

    def set_state_estimate_valid(self, valid) -> None:
        self.state_estimate_valid = valid

    def set_handshake_finished(self) -> None:
        self.handshake_finished = True

    def can_execute(self, entity) -> bool:
        return True if self.handshake_finished and self.state_estimate_valid else False

    def beginning_execution(self, entity) -> bool:
        self.state_estimate_valid = False
        return True

    def ending_execution(self, entity) -> None:
        pass
