import carla
import cv2
import numpy as np

VEHICLE_BOX_SIZE = (2.433, 1.5)  # length, width  --  x, y
MAP_ORIGIN = (-130, -80)
MAP_RESOLUTION = 0.1

class DrivableAreaMap(object):
    """ Carla world use left-hand coordinate system
        ---------> x
        |   -----------------------
        |   |                     |
        |   |                     |
        |   |                     |
        v y |                     |
            |                     |
            -----------------------
    """
    def __init__(self, map_file, map_origin=MAP_ORIGIN, resolution=MAP_RESOLUTION):
        self.map = cv2.imread(map_file)
        self.origin = np.array(map_origin)  # (x, y)
        self.resolution = resolution


class Vehicle(object):
    """ Carla world use left-hand coordinate system, with x pointing forward, y pointing left,
    z pointing up.
                    length
            -----------------------
            |                     |
            |                     |
            |           ---->x    | width
            |          |          |
            |          v y        |
            -----------------------
    """
    def __init__(self, pose, box_size=VEHICLE_BOX_SIZE):
        if isinstance(pose, carla.Transform):
            self.pose_mat = np.array(pose.get_matrix())
        elif isinstance(pose, np.ndarray):
            assert pose.shape == (4, 4)
            self.pose_mat = pose
        self.box_size = box_size

    def set_pose(self, pose):
        if isinstance(pose, carla.Transform):
            self.pose_mat = np.array(pose.get_matrix())
        elif isinstance(pose, np.ndarray):
            assert pose.shape == (4, 4)
            self.pose_mat = pose

    def get_box_grid(self, max_grid_size=MAP_RESOLUTION):
        num_grid_x = int(self.box_size[0] / max_grid_size)
        if max_grid_size * num_grid_x < self.box_size[0]:
            num_grid_x += 1

        num_grid_y = int(self.box_size[1] / max_grid_size)
        if max_grid_size * num_grid_y < self.box_size[1]:
            num_grid_y += 1

        xs = np.linspace(-self.box_size[0] / 2, self.box_size[0] / 2, num_grid_x + 1)
        ys = np.linspace(-self.box_size[1] / 2, self.box_size[1] / 2, num_grid_y + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_z = np.zeros_like(grid_x)
        grid = np.stack([grid_x, grid_y, grid_z], axis=-1)  # (num_grid_y, num_grid_x, 3)
        return grid @ self.pose_mat[:3, :3].T + self.pose_mat[:3, 3].T

    def in_drivable_area(self, drive_area):
        grid = self.get_box_grid(max_grid_size=drive_area.resolution)
        grid = grid[:, :, :2]  # (x, y)
        grid = (grid - drive_area.origin) / drive_area.resolution
        if ((grid < 0).any()
            or (grid[:, :, 0] >= drive_area.map.shape[1]).any()
            or (grid[:, :, 1] >= drive_area.map.shape[0]).any()):
            return False

        grid = grid.astype(np.int32)
        sampled_map = drive_area.map[grid[:, :, 1], grid[:, :, 0]]  # (num_grid_y, num_grid_x, 3)

        drive_area.map[grid[:, :, 1], grid[:, :, 0]] = (0, 0, 255)
        # TODO: return status


def test_vehicle_in_drivable_area():
    world = carla.Client('localhost', 2000).get_world()
    map_file = 'test_data/drivable_area_cv2.png'
    drive_area = DrivableAreaMap(map_file)

    spawn_points = world.get_map().get_spawn_points()
    for point in spawn_points:
        vehicle = Vehicle(point)
        in_drivable_area = vehicle.in_drivable_area(drive_area)
    cv2.imwrite('map.png', drive_area.map)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_vehicle_in_drivable_area()