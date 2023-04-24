import carla


cli = carla.Client('localhost', 2000)
world = cli.get_world()
carla_map = world.get_map()
opendrive_map = carla_map.to_opendrive()
map_name = carla_map.name.split('/')[-1]
with open(f'{map_name}.xodr', 'w') as f:
    f.write(opendrive_map)

import ipdb; ipdb.set_trace()
