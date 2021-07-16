import matplotlib
import matplotlib.pyplot
import numpy as np
import pathlib
import re

from vstdlib_random import UniformRandomStream

pathlib.Path('output').mkdir(exist_ok=True)


weapon_recoil_suppression_shots = np.int32(4)
weapon_recoil_suppression_factor = np.float32(0.75)
weapon_recoil_variance = np.float32(0.55)

TICK_INTERVAL = np.float32(1 / 64)
weapon_recoil_decay2_exp = np.int32(8)
weapon_recoil_decay2_lin = np.int32(18)
weapon_recoil_vel_decay = np.float32(4.5)

subdict = re.compile('\t*"([^\t|^"]+)"\n')
open_dict = re.compile('\t*{\n')
close_dict = re.compile('\t*}\n')
key_value = re.compile('\t*"(.+)"\t+"(.+)"\n')


def parse_items_game():
  data = {}
  path = [data]
  with open('items_game.txt') as file:
    for line in file:
      if result := subdict.fullmatch(line):
        assert open_dict.fullmatch(next(file))
        if result.group(1) not in path[-1]:
          path[-1][result.group(1)] = {}
        path.append(path[-1][result.group(1)])
      elif result := key_value.fullmatch(line):
        path[-1][result.group(1)] = result.group(2)
      elif close_dict.fullmatch(line):
        del path[-1]
      else:
        raise RuntimeError
  return data


def render(points, filepath):
  x_coordinates = [x for x, y in points[:]]
  y_coordinates = [y for x, y in points[:]]
  
  matplotlib.pyplot.clf()
  matplotlib.pyplot.plot(x_coordinates, y_coordinates)
  matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
  matplotlib.pyplot.axis('off')
  matplotlib.pyplot.savefig(filepath, dpi=500)


def rotate(points):
  cos_90 = np.cos(np.radians(-np.float32(90)))
  sin_90 = np.sin(np.radians(-np.float32(90)))
  for index, (x, y) in enumerate(points):
    points[index][0] = cos_90 * x - sin_90 * y
    points[index][1] = sin_90 * x + cos_90 * y
  return points


def invert(points):
  points[:,0] *= -1
  return points


def generate_recoil_table(attributes):
  recoil_table = np.empty((64, 2), dtype=np.float32)

  recoil_seed = np.int32(attributes['recoil seed'])
  full_auto = np.bool(attributes['is full auto'])

  recoil_angle = np.float32(attributes['recoil angle']) \
                 if 'recoil angle' in attributes else np.float32(0)

  recoil_magnitude = np.float32(attributes['recoil magnitude'])
  recoil_angle_variance = np.float32(attributes['recoil angle variance'])
  recoil_magnitude_variance = np.float32(attributes['recoil magnitude variance'])

  rng = UniformRandomStream(recoil_seed)
  angle, magnitude = np.float32(), np.float32()

  for i in range(len(recoil_table)):
    new_angle = recoil_angle + rng.random_float(
                  -recoil_angle_variance, recoil_angle_variance)
    new_magnitude = recoil_magnitude + rng.random_float(
                  -recoil_magnitude_variance, recoil_magnitude_variance)

    if full_auto and i > 0:
      angle = angle + (new_angle - angle) * weapon_recoil_variance
      magnitude = magnitude + (new_magnitude - magnitude) * weapon_recoil_variance
    else:
      angle = new_angle
      magnitude = new_magnitude

    if full_auto and i < weapon_recoil_suppression_shots:
      magnitude *= (weapon_recoil_suppression_factor
                      + (np.float32(1) - weapon_recoil_suppression_factor)
                      * np.float32(i / weapon_recoil_suppression_shots))

    recoil_table[i][0] = angle
    recoil_table[i][1] = magnitude
  return recoil_table


def get_angle_velocity(angle: np.float32, magnitude: np.float32):
  assert isinstance(angle, np.float32) and isinstance(magnitude, np.float32)
  pitch = -np.cos(np.radians(angle)) * magnitude
  yaw = -np.sin(np.radians(angle)) * magnitude
  return (pitch, yaw)


def hybrid_decay(aim_punch_angle):
  exp = weapon_recoil_decay2_exp * TICK_INTERVAL
  lin = weapon_recoil_decay2_lin * TICK_INTERVAL

  aim_punch_angle *= np.exp(-exp)

  mag = np.sqrt(aim_punch_angle[0] ** 2 + aim_punch_angle[1] ** 2)
  if mag > lin:
    aim_punch_angle *= (np.float32(1) - np.float32(lin / mag))
  else:
    aim_punch_angle.fill(np.float32())


def decay_aim_punch_angle(aim_punch_angle, aim_punch_velocity):
  hybrid_decay(aim_punch_angle)

  aim_punch_angle += aim_punch_velocity * TICK_INTERVAL * np.float32(0.5)
  aim_punch_velocity *= np.exp(TICK_INTERVAL * -weapon_recoil_vel_decay)
  aim_punch_angle += aim_punch_velocity * TICK_INTERVAL * np.float32(0.5)


def run():
  for key, value in parse_items_game()['items_game']['prefabs'].items():
    if 'prefab' in value and value['prefab'] in ('rifle', 'smg', 'machinegun'):
      print(f'processing {key}')
      recoil_table = generate_recoil_table(value['attributes'])
      current_time = np.float32()
      next_attack = np.float32()

      clip_size = int(value['attributes']['primary clip size'])
      shots = np.zeros((clip_size, 2), dtype=np.float32)
      aim_punch_angle = np.zeros(2, dtype=np.float32)
      aim_punch_velocity = np.zeros(2, dtype=np.float32)

      for index, (angle, magnitude) in enumerate(recoil_table[:clip_size]):
        # KickBack
        aim_punch_velocity += get_angle_velocity(angle, magnitude)

        current_attack = next_attack
        delta_attack = current_time - current_attack
        if delta_attack < 0 or delta_attack > TICK_INTERVAL:
          current_attack = current_time
        next_attack = current_attack + np.float32(value['attributes']['cycletime'])

        shots[index][0], shots[index][1] = aim_punch_angle[0], aim_punch_angle[1]
        while current_time < next_attack:
          decay_aim_punch_angle(aim_punch_angle, aim_punch_velocity)
          current_time += TICK_INTERVAL

      render(invert(rotate(shots)), f'output\\{key}.png')


if __name__ == '__main__':
  run()