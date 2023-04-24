from imap.lib.convertor import Opendrive2Apollo
from imap.lib.opendrive.map import Map
from imap import global_var
import matplotlib.pyplot as plt
import numpy as np

lane_type_to_color = {
    'driving': 'lightcoral',
    'shoulder': 'black',
    'sidewalk': 'skyblue',
    'median': 'springgreen',
    'none': 'yellow',
}

def draw_lanes(xodr_map: Map):
    plt.clf()
    for road_id, road in xodr_map.roads.items():
        road.generate_reference_line()
        road.add_offset_to_reference_line()
        road.add_origin_to_reference_line(0.0, 0.0)
        road.process_lanes()
        for lane_section in road.lanes.lane_sections:
            for lane in lane_section.left + lane_section.right:
                # if lane.lane_type != 'driving':
                #     continue
                left_x = [p.x for p in lane.left_boundary]
                left_y = [p.y for p in lane.left_boundary]
                right_x = [p.x for p in lane.right_boundary]
                right_y = [p.y for p in lane.right_boundary]
                polygon_x = left_x + right_x[::-1] + left_x[0:1]
                polygon_y = left_y + right_y[::-1] + left_y[0:1]
                plt.fill(polygon_x, polygon_y, c=lane_type_to_color[lane.lane_type])
                plt.plot(polygon_x, polygon_y, c='gray')
                print(road_id, '---', lane.lane_id, '---', lane.lane_type, '---', len(lane.left_boundary))
    # xs = np.arange(-150, 150, 10)
    # ys = np.arange(-150, 80, 10)
    # plt.xticks(xs)
    # plt.yticks(ys)
    for lane_type, color in lane_type_to_color.items():
        plt.scatter([], [], c=color, label=lane_type)
    plt.legend()
    plt.grid(True)
    plt.show()
    import ipdb; ipdb.set_trace()
    


if __name__ == '__main__':
    global_var._init()
    global_var.set_element_vaule('sampling_length', 0.1)

    path = '/home/robot/hongyu/ws/carla-export-data/Town10HD_Opt.xodr'
    xodr_map = Map()
    xodr_map.load(path)
    opendrive2apollo = Opendrive2Apollo(path)
    opendrive2apollo.set_parameters(False)
    # opendrive2apollo.convert()

    draw_lanes(xodr_map)
