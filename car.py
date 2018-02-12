import numpy as np
import pygame
import copy
import random


class Car:
    def __init__(self, pos, map, direction):
        self.max_speed = 600.0
        self.min_speed = 10.0
        self.max_turning = np.radians(90.0)

        self.color = (190, 220, 170)
        self.reward_font = pygame.font.Font(None, 16)
        self.draw_radius = 8.0
        self.draw_radius_crashed = 6.0

        self.crashed = False
        self.map = map

        self.start_pos = np.array(pos, dtype=np.float32)
        self.start_direction = direction
        self.start_speed = self.min_speed
        self.reset()

    def reset(self):
        self.pos = self.start_pos.copy()
        self.speed = self.start_speed
        self.direction = self.start_direction
        self.total_distance = 0.0

    def move(self, seconds_passed):
        if self.crashed:
            return
        # Update position.
        distance = self.speed * seconds_passed
        self.pos += self.direction * distance
        self.total_distance += distance

        # Dampen the speed.
        self.speed = 0.98 * self.speed

        # Check for crash.
        if self.map.is_colliding(self.pos):
            self.crashed = True

    def draw(self, screen):
        radius = self.draw_radius if not self.crashed else self.draw_radius_crashed
        pygame.draw.circle(screen, self.color, self.pos, int(radius))
        reward = self.reward_font.render(str(int(self.total_distance)), True, (255, 255, 0))
        screen.blit(reward, (self.pos[0] - reward.get_width() - self.draw_radius, self.pos[1] - 0.5 * reward.get_height()))

    def steer(self, acceleration, direction, seconds_passed):
        if self.crashed:
            return
        theta = direction * seconds_passed * self.max_turning
        c, s = np.cos(theta), np.sin(theta)
        self.direction = np.array([[c, -s], [s, c]]).dot(self.direction)
        if acceleration > 0.0:
            self.speed += acceleration * 100.0 * seconds_passed
        else:
            self.speed += acceleration * 200.0 * seconds_passed
        if self.speed < self.min_speed:
            self.speed = self.min_speed
        if self.speed > self.max_speed:
            self.speed = self.max_speed


class SensorCar(Car):
    def __init__(self, pos, map):
        super(SensorCar, self).__init__(pos, map, np.array([1.0, 0.0], dtype=np.float32))
        self.sensor_range = 100.0
        self.sensor_count = 5
        self.sensor_fov = np.radians(90.0)

    def get_sensor_directions(self):
        sensor_directions = []
        theta = - 0.5 * self.sensor_fov
        theta_step = self.sensor_fov / (self.sensor_count - 1)
        for i in range(self.sensor_count):
            c, s = np.cos(theta + i * theta_step), np.sin(theta + i * theta_step)
            direction = np.array([[c, -s], [s, c]]).dot(self.direction)
            sensor_directions.append(direction)
        return sensor_directions

    def read_sensors(self):
        reading = np.full((self.sensor_count + 1, 1), self.sensor_range, dtype=np.float32)  # Init with maximum reading.
        for i, s in enumerate(self.get_sensor_directions()):
            for d in range(int(self.sensor_range)):
                scan_pos = d * s + self.pos
                if self.map.is_colliding(scan_pos.astype(np.int)):
                    reading[i, 0] = d
                    break
        reading[-1, 0] = self.speed
        return reading


    def draw(self, screen):
        if not self.crashed:
            for s in self.get_sensor_directions():
                pygame.draw.line(screen, (90, 90, 90), self.pos, self.pos + s * self.sensor_range, 1)
            pygame.draw.line(screen, self.color, self.pos, self.pos + self.speed * self.direction, 2)
        super(SensorCar, self).draw(screen)


class SelfDrivingCar(SensorCar):
    def __init__(self, pos, map, copy_from=None):
        super(SelfDrivingCar, self).__init__(pos, map)

        def random_array(rows, cols):
            return 2.0 * np.random.rand(rows, cols) - 1.0

        if copy_from:
            self.fc1_cnt = copy_from.fc1_cnt
            self.fc1_w = copy_from.fc1_w.copy()
            self.fc1_b = copy_from.fc1_b.copy()

            self.fc2_cnt = copy_from.fc2_cnt
            self.fc2_w = copy_from.fc2_w.copy()
            self.fc2_b = copy_from.fc2_b.copy()

            self.fc_out_cnt = copy_from.fc_out_cnt
            self.fc_out_w = copy_from.fc_out_w.copy()
            self.fc_out_b = copy_from.fc_out_b.copy()
        else:
            self.fc1_cnt = self.sensor_count - 1
            self.fc1_w = random_array(self.fc1_cnt, self.sensor_count + 1)
            self.fc1_b = random_array(self.fc1_cnt, 1)

            self.fc2_cnt = self.sensor_count - 1
            self.fc2_w = random_array(self.fc2_cnt, self.fc1_cnt)
            self.fc2_b = random_array(self.fc2_cnt, 1)

            self.fc_out_cnt = 2
            self.fc_out_w = random_array(self.fc_out_cnt, self.fc2_cnt)
            self.fc_out_b = random_array(self.fc_out_cnt, 1)

    def update(self, seconds_passed):
        sensor_readings = self.read_sensors()

        fc1 = np.tanh(self.fc1_w.dot(sensor_readings) + self.fc1_b)
        fc2 = np.tanh(self.fc2_w.dot(fc1) + self.fc2_b)
        out = np.tanh(self.fc_out_w.dot(fc2) + self.fc_out_b)

        self.steer(out[0,0], out[1, 0], seconds_passed)


def crossover(a, b):
    child = SelfDrivingCar(a.start_pos, a.map, copy_from=a)

    def breed(array_a, array_b, by_column):
        ret = array_a.copy()
        split = np.random.randint(1, (ret.shape[1] if by_column else ret.shape[0]) - 1)
        if by_column:
            if bool(random.getrandbits(1)):
                ret[:, 0:split] = array_b[:, 0:split].copy()
            else:
                ret[:, split:] = array_b[:, split:].copy()
        else:
            if bool(random.getrandbits(1)):
                ret[0:split, :] = array_b[0:split, :].copy()
            else:
                ret[split:, :] = array_b[split:, :].copy()
        return ret

    child.fc1_w = breed(a.fc1_w, b.fc1_w, True)
    child.fc2_w = breed(a.fc2_w, b.fc2_w, True)
    child.fc_out_w = breed(a.fc_out_w, b.fc_out_w, True)

    child.fc1_b = breed(a.fc1_b, b.fc1_b, False)
    child.fc2_b = breed(a.fc2_b, b.fc2_b, False)

    # These are only of shape (2, 1)
    if bool(random.getrandbits(1)):
        if bool(random.getrandbits(1)):
            child.fc_out_b[0] = b.fc_out_b[0].copy()
        else:
            child.fc_out_b[1] = b.fc_out_b[1].copy()
    else:
        if bool(random.getrandbits(1)):
            child.fc_out_b = b.fc_out_b.copy()

    return child

def mutation(car):
    mutated = SelfDrivingCar(car.start_pos, car.map, copy_from=car)
    def mutate(array, amt):
        for r in range(array.shape[0]):
            if random.getrandbits(1):
                pos = np.random.randint(0, array.shape[1])
                std = max(np.std(array[r, :]), 0.1)
                array[r, pos] += amt * std * (2.0 * np.random.ranf() - 1.0)  # Random in the range [-std, std).
    mutate(mutated.fc1_w, .5)
    mutate(mutated.fc2_w, .5)
    mutate(mutated.fc_out_w, .5)
    mutate(mutated.fc1_b, .1)
    mutate(mutated.fc2_b, .1)
    mutate(mutated.fc_out_b, .1)
    return mutated

