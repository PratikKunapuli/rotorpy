from rotorpy.utils.occupancy_map import OccupancyMap
from rotorpy.world import World
import numpy as np  # For array creation/manipulation
from scipy.spatial.transform import Rotation as R


world_size = 10
vavg = 2
yaw_min = -0.85 * np.pi
yaw_max = 0.85 * np.pi

# world_buffer = 2
# min_distance = 1
# max_distance = min_distance + 3

world = World.empty(
        [
            -world_size / 2,
            world_size / 2,
            -world_size / 2,
            world_size / 2,
            -world_size / 2,
            world_size / 2,
        ]
    )

# world = World.empty(
#         [
#             0,
#             2,
#             4.3,
#             6.3,
#             0.8,
#             2.8,
#         ]
#     )


def sample_waypoints(
    num_waypoints=4,
    world=world,
    world_buffer=2,
    check_collision=True,
    min_distance=1,
    max_distance=3,
    max_attempts=1000,
    start_waypoint=None,
    end_waypoint=None,
    rng=None,
    seed=None,
):
    """
    Samples random waypoints (x,y,z) in the world. Ensures waypoints do not collide with objects, although there is no guarantee that
    the path you generate with these waypoints will be collision free.
    Inputs:
        num_waypoints: Number of waypoints to sample.
        world: Instance of World class containing the map extents and any obstacles.
        world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
            from the edge of the world.
        check_collision: If True, checks for collisions with obstacles. If False, does not check for collisions. Checking collisions slows down the script.
        min_distance: Minimum distance between waypoints consecutive waypoints.
        max_distance: Maximum distance between consecutive waypoints.
        max_attempts: Maximum number of attempts to sample a waypoint.
        start_waypoint: If specified, the first waypoint will be this point.
        end_waypoint: If specified, the last waypoint will be this point.
    Outputs:
        waypoints: A list of (x,y,z) waypoints. [[waypoint_1], [waypoint_2], ... , [waypoint_n]]
    """

    if min_distance > max_distance:
        raise Exception("min_distance must be less than or equal to max_distance.")

    def check_distance(waypoint, waypoints, min_distance, max_distance):
        """
        Checks if the waypoint is at least min_distance away from all other waypoints.
        Inputs:
            waypoint: The waypoint to check.
            waypoints: A list of waypoints.
            min_distance: The minimum distance the waypoint must be from all other waypoints.
            max_distance: The maximum distance the waypoint can be from all other waypoints.
        Outputs:
            collision: True if the waypoint is at least min_distance away from all other waypoints. False otherwise.
        """
        collision = False
        for w in waypoints:
            if (np.linalg.norm(waypoint - w) < min_distance) or (
                np.linalg.norm(waypoint - w) > max_distance
            ):
                collision = True
        return collision

    def check_obstacles(waypoint, occupancy_map):
        """
        Checks if the waypoint is colliding with any obstacles in the world.
        Inputs:
            waypoint: The waypoint to check.
            occupancy_map: An instance of the occupancy map.
        Outputs:
            collision: True if the waypoint is colliding with any obstacles in the world. False otherwise.
        """
        collision = False
        if occupancy_map.is_occupied_metric(waypoint):
            collision = True
        return collision

    def single_sample(
        world,
        current_waypoints,
        world_buffer,
        occupancy_map,
        min_distance,
        max_distance,
        max_attempts=1000,
        rng=None,
        seed=None,
    ):
        """
        Samples a single waypoint.
        Inputs:
            world: Instance of World class containing the map extents and any obstacles.
            world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
                from the edge of the world.
            occupancy_map: An instance of the occupancy map.
            min_distance: Minimum distance between waypoints consecutive waypoints.
            max_distance: Maximum distance between consecutive waypoints.
            max_attempts: Maximum number of attempts to sample a waypoint.
            rng: Random number generator. If None, uses numpy's random number generator.
            seed: Seed for the random number generator.
        Outputs:
            waypoint: A single (x,y,z) waypoint.
        """

        if seed is not None:
            np.random.seed(seed)

        num_attempts = 0

        world_lower_limits = (
            np.array(world.world["bounds"]["extents"][0::2]) + world_buffer
        )
        world_upper_limits = (
            np.array(world.world["bounds"]["extents"][1::2]) - world_buffer
        )

        if len(current_waypoints) == 0:
            max_distance_lower_limits = world_lower_limits
            max_distance_upper_limits = world_upper_limits
        else:
            max_distance_lower_limits = current_waypoints[-1] - max_distance
            max_distance_upper_limits = current_waypoints[-1] + max_distance

        lower_limits = np.max(
            np.vstack((world_lower_limits, max_distance_lower_limits)), axis=0
        )
        upper_limits = np.min(
            np.vstack((world_upper_limits, max_distance_upper_limits)), axis=0
        )

        waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
        while check_obstacles(waypoint, occupancy_map) or (
            check_distance(waypoint, current_waypoints, min_distance, max_distance)
            if occupancy_map is not None
            else False
        ):
            waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
            num_attempts += 1
            if num_attempts > max_attempts:
                raise Exception(
                    "Could not sample a waypoint after {} attempts. Issue with obstacles: {}, Issue with min/max distance: {}".format(
                        max_attempts,
                        check_obstacles(waypoint, occupancy_map),
                        check_distance(
                            waypoint, current_waypoints, min_distance, max_distance
                        ),
                    )
                )
        return waypoint

    ######################################################################################################################

    waypoints = []

    if check_collision:
        # Create occupancy map from the world. This can potentially be slow, so only do it if the user wants to check for collisions.
        occupancy_map = OccupancyMap(
            world=world, resolution=[0.5, 0.5, 0.5], margin=0.1
        )
    else:
        occupancy_map = None

    if start_waypoint is not None:
        waypoints = [start_waypoint]
    else:
        # Randomly sample a start waypoint.
        waypoints.append(
            single_sample(
                world,
                waypoints,
                world_buffer,
                occupancy_map,
                min_distance,
                max_distance,
                max_attempts,
                rng,
                seed,
            )
        )

    num_waypoints -= 1

    if end_waypoint is not None:
        num_waypoints -= 1

    for _ in range(num_waypoints):
        waypoints.append(
            single_sample(
                world,
                waypoints,
                world_buffer,
                occupancy_map,
                min_distance,
                max_distance,
                max_attempts,
                rng,
                seed,
            )
        )

    if end_waypoint is not None:
        waypoints.append(end_waypoint)

    return np.array(waypoints)


# def sample_waypoints(
#     num_waypoints = 4,
#     world = World.empty(
#         [
#             -world_size / 2,
#             world_size / 2,
#             -world_size / 2,
#             world_size / 2,
#             -world_size / 2,
#             world_size / 2,
#         ]
#     ),
#     world_buffer = 2,
#     check_collision = True,
#     min_distance = 1,
#     max_distance = 4,
#     max_attempts = 1000,
#     start_waypoint = None,
#     end_waypoint = None,
# ):
#     """
#     Samples random waypoints (x,y,z) in the world. Ensures waypoints do not collide with objects, although there is no guarantee that
#     the path you generate with these waypoints will be collision free.
#     Inputs:
#         num_waypoints: Number of waypoints to sample.
#         world: Instance of World class containing the map extents and any obstacles.
#         world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
#             from the edge of the world.
#         check_collision: If True, checks for collisions with obstacles. If False, does not check for collisions. Checking collisions slows down the script.
#         min_distance: Minimum distance between waypoints consecutive waypoints.
#         max_distance: Maximum distance between consecutive waypoints.
#         max_attempts: Maximum number of attempts to sample a waypoint.
#         start_waypoint: If specified, the first waypoint will be this point.
#         end_waypoint: If specified, the last waypoint will be this point.
#     Outputs:
#         waypoints: A list of (x,y,z) waypoints. [[waypoint_1], [waypoint_2], ... , [waypoint_n]]
#     """

#     if min_distance > max_distance:
#         raise Exception("min_distance must be less than or equal to max_distance.")

#     def check_distance(waypoint, waypoints, min_distance, max_distance):
#         """
#         Checks if the waypoint is at least min_distance away from all other waypoints.
#         Inputs:
#             waypoint: The waypoint to check.
#             waypoints: A list of waypoints.
#             min_distance: The minimum distance the waypoint must be from all other waypoints.
#             max_distance: The maximum distance the waypoint can be from all other waypoints.
#         Outputs:
#             collision: True if the waypoint is at least min_distance away from all other waypoints. False otherwise.
#         """
#         collision = False
#         for w in waypoints:
#             if (np.linalg.norm(waypoint - w) < min_distance) or (
#                 np.linalg.norm(waypoint - w) > max_distance
#             ):
#                 collision = True
#         return collision

#     def check_obstacles(waypoint, occupancy_map):
#         """
#         Checks if the waypoint is colliding with any obstacles in the world.
#         Inputs:
#             waypoint: The waypoint to check.
#             occupancy_map: An instance of the occupancy map.
#         Outputs:
#             collision: True if the waypoint is colliding with any obstacles in the world. False otherwise.
#         """
#         collision = False
#         if occupancy_map.is_occupied_metric(waypoint):
#             collision = True
#         return collision

#     def single_sample(
#         world,
#         current_waypoints,
#         world_buffer,
#         occupancy_map,
#         min_distance,
#         max_distance,
#         max_attempts=1000,
#         rng=None,
#     ):
#         """
#         Samples a single waypoint.
#         Inputs:
#             world: Instance of World class containing the map extents and any obstacles.
#             world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
#                 from the edge of the world.
#             occupancy_map: An instance of the occupancy map.
#             min_distance: Minimum distance between waypoints consecutive waypoints.
#             max_distance: Maximum distance between consecutive waypoints.
#             max_attempts: Maximum number of attempts to sample a waypoint.
#             rng: Random number generator. If None, uses numpy's random number generator.
#         Outputs:
#             waypoint: A single (x,y,z) waypoint.
#         """

#         num_attempts = 0

#         world_lower_limits = (
#             np.array(world.world["bounds"]["extents"][0::2]) + world_buffer
#         )
#         world_upper_limits = (
#             np.array(world.world["bounds"]["extents"][1::2]) - world_buffer
#         )

#         if len(current_waypoints) == 0:
#             max_distance_lower_limits = world_lower_limits
#             max_distance_upper_limits = world_upper_limits
#         else:
#             max_distance_lower_limits = current_waypoints[-1] - max_distance
#             max_distance_upper_limits = current_waypoints[-1] + max_distance

#         lower_limits = np.max(
#             np.vstack((world_lower_limits, max_distance_lower_limits)), axis=0
#         )
#         upper_limits = np.min(
#             np.vstack((world_upper_limits, max_distance_upper_limits)), axis=0
#         )

#         waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
#         while check_obstacles(waypoint, occupancy_map) or (
#             check_distance(waypoint, current_waypoints, min_distance, max_distance)
#             if occupancy_map is not None
#             else False
#         ):
#             waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
#             num_attempts += 1
#             if num_attempts > max_attempts:
#                 raise Exception(
#                     "Could not sample a waypoint after {} attempts. Issue with obstacles: {}, Issue with min/max distance: {}".format(
#                         max_attempts,
#                         check_obstacles(waypoint, occupancy_map),
#                         check_distance(
#                             waypoint, current_waypoints, min_distance, max_distance
#                         ),
#                     )
#                 )
#         return waypoint

#     ######################################################################################################################

#     waypoints = []

#     if check_collision:
#         # Create occupancy map from the world. This can potentially be slow, so only do it if the user wants to check for collisions.
#         occupancy_map = OccupancyMap(
#             world=world, resolution=[0.5, 0.5, 0.5], margin=0.1
#         )
#     else:
#         occupancy_map = None

#     if start_waypoint is not None:
#         waypoints = [start_waypoint]
#     else:
#         # Randomly sample a start waypoint.
#         waypoints.append(
#             single_sample(
#                 world,
#                 waypoints,
#                 world_buffer,
#                 occupancy_map,
#                 min_distance,
#                 max_distance,
#                 max_attempts,
#             )
#         )

#     num_waypoints -= 1

#     if end_waypoint is not None:
#         num_waypoints -= 1

#     for _ in range(num_waypoints):
#         waypoints.append(
#             single_sample(
#                 world,
#                 waypoints,
#                 world_buffer,
#                 occupancy_map,
#                 min_distance,
#                 max_distance,
#                 max_attempts,
#             )
#         )

#     if end_waypoint is not None:
#         waypoints.append(end_waypoint)

#     return np.array(waypoints)


def compute_yaw_from_quaternion(quaternion):
    R_matrix = R.from_quat(quaternion).as_matrix()
    b3 = R_matrix[:, 2]
    H = np.zeros((3, 3))
    H[:, :] = np.array(
            [
                [
                    1 - (b3[0] ** 2) / (1 + b3[2]),
                    -(b3[0] * b3[1]) / (1 + b3[2]),
                    b3[0],
                ],
                [
                    -(b3[0] * b3[1]) / (1 + b3[2]),
                    1 - (b3[1] ** 2) / (1 + b3[2]),
                    b3[1],
                ],
                [-b3[0], -b3[1], b3[2]],
            ]
    )
    Hyaw = np.transpose(H) @ R_matrix
    actual_yaw = np.arctan2(Hyaw[1, 0], Hyaw[0, 0])
    return actual_yaw