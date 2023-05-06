import cv2
from imap.lib.convertor import Opendrive2Apollo
from imap.lib.opendrive.map import Map
from imap import global_var
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from tqdm import tqdm


LANE_TYPE_TO_COLOR = {
    'driving': 'lightcoral',
    'shoulder': 'black',
    'sidewalk': 'skyblue',
    'median': 'springgreen',
    'none': 'yellow',
}
VEHICLE_BOX_SIZE = (2.433, 1.5)  # length, width
MAP_ORIGIN_ODR = (-130, 80)  # x, y (in odr coordinate)
MAP_ORIGIN_CARLA = (MAP_ORIGIN_ODR[0], -MAP_ORIGIN_ODR[1])  # x, y (in carla coordinate)
MAP_RANGE = ((MAP_ORIGIN_CARLA[0], MAP_ORIGIN_CARLA[0] + 250),
             (MAP_ORIGIN_CARLA[1], MAP_ORIGIN_CARLA[1] + 250))  # ((min_x, max_x), (min_y, max_y))  in carla coordinate


def process_xodr_map(xodr_map: Map):
    print('Adding offsets to roads...')
    for road_id, road in tqdm(xodr_map.roads.items()):
        assert not road.reference_line
        road.generate_reference_line()
        road.add_offset_to_reference_line()
        road.add_origin_to_reference_line(0.0, 0.0)
        road.process_lanes()
    return xodr_map


def draw_lanes(xodr_map: Map, invert_y=True):
    plt.clf()
    if invert_y:
        plt.gca().invert_yaxis()

    polygons = {key: [] for key in LANE_TYPE_TO_COLOR}
    for road_id, road in xodr_map.roads.items():
        assert road.reference_line
        for lane_section in road.lanes.lane_sections:
            for lane in lane_section.left + lane_section.right:
                # if lane.lane_type != 'driving':
                #     continue
                left_x = [p.x for p in lane.left_boundary]
                left_y = [p.y for p in lane.left_boundary]
                right_x = [p.x for p in lane.right_boundary]
                right_y = [p.y for p in lane.right_boundary]
                polygon_x = np.array(left_x + right_x[::-1] + left_x[0:1])
                polygon_y = np.array(left_y + right_y[::-1] + left_y[0:1])
                if invert_y:
                    polygon_y = -polygon_y
                polygons[lane.lane_type].append((polygon_x, polygon_y))
                # print(road_id, '---', lane.lane_id, '---', lane.lane_type, '---', len(lane.left_boundary))
    
    print('Drawing polygons...')
    for lane_type in ['none', 'median', 'shoulder', 'sidewalk', 'driving']:
        for polygon_x, polygon_y in polygons[lane_type]:
            plt.fill(polygon_x, polygon_y, c=LANE_TYPE_TO_COLOR[lane_type])
    for lane_type in ['none', 'median', 'shoulder', 'sidewalk', 'driving']:
        for polygon_x, polygon_y in polygons[lane_type]:
            plt.plot(polygon_x, polygon_y, c='gray')

    # xs = np.arange(-150, 150, 10)
    # ys = np.arange(-150, 80, 10)
    # plt.xticks(xs)
    # plt.yticks(ys)
    for lane_type, color in LANE_TYPE_TO_COLOR.items():
        plt.scatter([], [], c=color, label=lane_type)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('test_data/drivable_area.png', dpi=300)


def draw_lanes_cv2(xodr_map: Map, resolution=0.1):
    canvas = np.ones((int((MAP_RANGE[1][1] - MAP_RANGE[1][0]) / resolution),
                      int((MAP_RANGE[0][1] - MAP_RANGE[0][0]) / resolution),
                      3), dtype=np.uint8) * 255
    polygons = {key: [] for key in LANE_TYPE_TO_COLOR}
    for road_id, road in xodr_map.roads.items():
        assert road.reference_line
        for lane_section in road.lanes.lane_sections:
            for lane in lane_section.left + lane_section.right:
                # if lane.lane_type != 'driving':
                #     continue
                left_x = [p.x for p in lane.left_boundary]
                left_y = [p.y for p in lane.left_boundary]
                right_x = [p.x for p in lane.right_boundary]
                right_y = [p.y for p in lane.right_boundary]
                polygon_x = np.array(left_x + right_x[::-1])
                polygon_y = np.array(left_y + right_y[::-1])
                polygon_y = -polygon_y  # invert y (odr coordinate to carla coordinate)
                polygon = np.stack([polygon_x, polygon_y], axis=1)
                pixels = ((polygon - MAP_ORIGIN_CARLA) / resolution).astype(np.int32)
                polygons[lane.lane_type].append(pixels)
                # print(road_id, '---', lane.lane_id, '---', lane.lane_type, '---', len(lane.left_boundary))
    
    print('Drawing polygons...')
    for lane_type in ['none', 'median', 'shoulder', 'sidewalk', 'driving']:
        color = (np.array(colors.to_rgb(LANE_TYPE_TO_COLOR[lane_type])) * 255).astype(
            np.int32).tolist()[::-1]
        for polygon in polygons[lane_type]:
            cv2.fillPoly(canvas, [polygon], color)
    all_polygons = sum(polygons.values(), start=[])
    cv2.polylines(canvas, all_polygons, True, (0, 0, 0), thickness=1)

    cv2.imwrite('test_data/drivable_area_cv2.png', canvas)
    return canvas


def draw_vehicle(transform_matrix, box_size=VEHICLE_BOX_SIZE):
    transform_matrix = np.array(transform_matrix)
    corners = np.array([
        [box_size[0] / 2, box_size[1] / 2, 0],
        [box_size[0] / 2, -box_size[1] / 2, 0],
        [-box_size[0] / 2, -box_size[1] / 2, 0],
        [-box_size[0] / 2, box_size[1] / 2, 0],
    ])
    corners = corners @ transform_matrix[:3, :3].T + transform_matrix[:3, 3].T
    plt.fill(corners[:, 0], corners[:, 1], c='red', alpha=0.5)
    # plt.savefig('test_data/drivable_area.png', dpi=300)


def draw_vehicle_cv2(canvas, transform_matrix, resolution=0.1, box_size=VEHICLE_BOX_SIZE):
    transform_matrix = np.array(transform_matrix)
    corners = np.array([
        [box_size[0] / 2, box_size[1] / 2, 0],
        [box_size[0] / 2, -box_size[1] / 2, 0],
        [-box_size[0] / 2, -box_size[1] / 2, 0],
        [-box_size[0] / 2, box_size[1] / 2, 0],
    ])
    corners = corners @ transform_matrix[:3, :3].T + transform_matrix[:3, 3].T
    pixels = ((corners[:, :2] - MAP_ORIGIN_CARLA) / resolution).astype(np.int32)
    cv2.fillPoly(canvas, [pixels], (0, 0, 255))


def draw_carla_spawn_points():
    import carla
    cli = carla.Client('localhost', 2000)
    world = cli.get_world()
    points = world.get_map().get_spawn_points()
    print('Drawing vehicles...')
    for point in tqdm(points):
        draw_vehicle(point.get_matrix())
    plt.savefig('test_data/drivable_area_spawn_points.png', dpi=300)


def draw_carla_spawn_points_cv2(canvas, resolution=0.1):
    import carla
    cli = carla.Client('localhost', 2000)
    world = cli.get_world()
    points = world.get_map().get_spawn_points()
    print('Drawing vehicles...')
    for point in tqdm(points):
        draw_vehicle_cv2(canvas, point.get_matrix(), resolution)
    cv2.imwrite('test_data/drivable_area_spawn_points_cv2.png', canvas)


if __name__ == '__main__':
    global_var._init()
    global_var.set_element_vaule('sampling_length', 0.1)

    path = 'test_data/Town10HD_Opt.xodr'
    # opendrive2apollo = Opendrive2Apollo(path)
    # opendrive2apollo.set_parameters(False)
    # opendrive2apollo.convert()

    xodr_map = Map()
    xodr_map.load(path)
    xodr_map = process_xodr_map(xodr_map)

    draw_lanes(xodr_map, invert_y=True)
    draw_carla_spawn_points()

    canvas = draw_lanes_cv2(xodr_map, resolution=0.1)
    draw_carla_spawn_points_cv2(canvas, resolution=0.1)