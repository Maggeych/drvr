import pygame
import numpy as np


class HexMap:
    def __init__(self, cell_distance, window_size):
        # Parameters.
        self.occupied_line_color = (40, 170, 30)
        self.occupied_fill_color = (60, 190, 50)
        self.empty_line_color = (120, 130, 100)
        self.empty_fill_color = (140, 150, 120)

        self.inner_radius = 0.5 * cell_distance
        self.vertical = True  # True for vertical hexagon orientation, False for horizontal.

        # Initialization.
        self.outer_radius = self.inner_radius / 0.866005404
        x_cnt = (window_size[0] + 1) / (2.0 * self.inner_radius) - 1
        y_cnt = (window_size[1] + 4) / (1.5 * self.outer_radius) - 1
        if x_cnt % 1 > 0.5:
            x_cnt += 1
        if y_cnt % 1 > 0.5:
            y_cnt += 1
        self.map = np.zeros((int(y_cnt), int(x_cnt)), dtype=np.bool)
        self.map[1, 1:-2] = True
        self.map[-2, 1:-2] = True
        self.map[2:-2, 1] = True
        self.map[2:-2, -2] = True

        self.collision_lookup = pygame.Surface(window_size)
        self.update_collision_lookup()

    def draw_hexagon(self, screen, pos, line, fill):
        vertices = [
            (pos[0], pos[1] + self.outer_radius),
            (pos[0] + self.inner_radius, pos[1] + 0.5 * self.outer_radius),
            (pos[0] + self.inner_radius, pos[1] - 0.5 * self.outer_radius),
            (pos[0], pos[1] - self.outer_radius),
            (pos[0] - self.inner_radius, pos[1] - 0.5 * self.outer_radius),
            (pos[0] - self.inner_radius, pos[1] + 0.5 * self.outer_radius),
            (pos[0], pos[1] + self.outer_radius)
        ]
        pygame.draw.polygon(screen, fill, vertices, 0)
        pygame.draw.polygon(screen, line, vertices, 2)

    def draw(self, screen):
        offset_x = self.inner_radius
        offset_y = self.outer_radius
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                cell_x = int(x * 2 * self.inner_radius + offset_x)
                cell_y = int(y * 1.5 * self.outer_radius + offset_y)
                if y % 2:
                    cell_x += self.inner_radius
                if not self.map[y, x]:
                    self.draw_hexagon(screen, (cell_x, cell_y), self.occupied_line_color, self.occupied_fill_color)
                else:
                    self.draw_hexagon(screen, (cell_x, cell_y), self.empty_line_color, self.empty_fill_color)

    def update_collision_lookup(self):
        self.collision_lookup.fill((0, 0, 0))
        offset_x = self.inner_radius
        offset_y = self.outer_radius
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                cell_x = int(x * 2 * self.inner_radius + offset_x)
                cell_y = int(y * 1.5 * self.outer_radius + offset_y)
                if y % 2:
                    cell_x += self.inner_radius
                if not self.map[y, x]:
                    self.draw_hexagon(self.collision_lookup, (cell_x, cell_y), (255, 255, 255), (255, 255, 255))

    def set_cell_at(self, screen_pos, value):
        idx_y = int((screen_pos[1] + 4) / (1.5 * self.outer_radius))
        if idx_y % 2:
            screen_pos = (screen_pos[0] - self.inner_radius, screen_pos[1])
        idx_x = int((screen_pos[0] + 1) / (2.0 * self.inner_radius))
        if self.map[int(idx_y), int(idx_x)] is not value:
            self.map[int(idx_y), int(idx_x)] = value
            self.update_collision_lookup()

    def is_colliding(self, pos):
        try:
            return self.collision_lookup.get_at(pos) == (255, 255, 255, 255)
        except IndexError:
            return True
