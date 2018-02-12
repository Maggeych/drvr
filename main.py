import pygame
import numpy as np
import car
import hexmap
import copy


class Game:
    def __init__(self):
        pygame.font.init()
        self.window_size = (800, 600)
        self.max_fps = 30
        self.nr_of_starting_cars = 50
        self.background_color = (60, 60, 60)

        self.clock = pygame.time.Clock()
        self.map = hexmap.HexMap(50, self.window_size)
        self.cars = []
        for i in range(self.nr_of_starting_cars):
            self.cars.append(car.SelfDrivingCar((80, 70), self.map))
        self.reset_cars()

    def reset_cars(self):
        for car in self.cars:
            car.reset()

    def run(self):
        screen = pygame.display.set_mode(self.window_size)
        stat_font = pygame.font.Font(None, 24)
        stat_color = (255, 255, 0)
        stat_pos = (self.window_size[0]-200, 20)

        running = True
        stop_trial = False
        total_time = 0.0
        generation_number = 1
        while running:
            # time_passed = self.clock.tick(self.max_fps) * .001
            time_passed = self.clock.tick() * .001
            total_time += time_passed

            # Handle events.
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    running = False
                elif event.type is pygame.KEYDOWN:
                    if event.key is pygame.K_ESCAPE:
                        running = False
                    elif event.key is pygame.K_r:
                        stop_trial = True
                elif event.type is pygame.MOUSEMOTION or event.type is pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    pressed = pygame.mouse.get_pressed()
                    if pressed[0]:  # Left mouse button.
                        self.map.set_cell_at(pos, True)
                    elif pressed[2]:  # Right mouse button.
                        self.map.set_cell_at(pos, False)

            # Update the objects.
            nr_cars_alive = 0
            for car in self.cars:
                car.move(time_passed)
                car.update(time_passed)
                if not car.crashed:
                    nr_cars_alive += 1

            # Draw the objects.
            screen.fill(self.background_color)

            self.map.draw(screen)

            for car in self.cars:
                car.draw(screen)

            fps = stat_font.render("FPS: %3.0f" % self.clock.get_fps(), True, stat_color)
            screen.blit(fps, stat_pos)
            time = stat_font.render("Time: %2.0f" % total_time, True, stat_color)
            screen.blit(time, (stat_pos[0], stat_pos[1] + stat_font.get_linesize()))
            gen = stat_font.render("Generation: %3d" % generation_number, True, stat_color)
            screen.blit(gen, (stat_pos[0], stat_pos[1] + 2 * stat_font.get_linesize()))
            alive = stat_font.render("Alive: %3d" % nr_cars_alive, True, stat_color)
            screen.blit(alive, (stat_pos[0], stat_pos[1] + 3 * stat_font.get_linesize()))

            pygame.display.flip()

            # Check if the current run is over.
            if total_time >= 20.0 or nr_cars_alive is 0 or stop_trial:
                stop_trial = False
                total_time = 0.0
                generation_number += 1
                self.evolve()

    def evolve(self):
        self.cars.sort(key=lambda car: car.total_distance)
        prob_dist = np.array([car.total_distance for car in self.cars], dtype=np.float32)
        prob_dist = prob_dist / np.sum(prob_dist)

        # Select parents.
        nr_of_parents = 5
        choices = np.random.choice(len(self.cars), nr_of_parents, p=prob_dist)
        parents = []
        for i in choices:
            clone = car.SelfDrivingCar(self.cars[i].start_pos, self.map, copy_from=self.cars[i])
            clone.color = (255, 0, 0)
            parents.append(clone)

        # Crossover.
        offspring = []
        for a in range(len(parents) - 1):
            for i in range(len(parents) - 1 - a):
                b = a + i
                off = car.crossover(parents[a], parents[b])
                offspring.append(off)

        self.cars = [self.cars[-1]] + parents + offspring

        # Mutations.
        nr_of_mutations = 10
        for i in range(nr_of_mutations):
            mutation = car.mutation(self.cars[np.random.randint(0, len(self.cars))])
            mutation.color = (255, 0, 255)
            self.cars.append(mutation)

        self.cars.reverse()
        self.reset_cars()


def get_command():
    # Use like so:
    # command = get_command()
    # self.car.steer(command[0], command[1], time_passed)

    # if np.any(self.car.pos > self.window_size) or np.any(self.car.pos < [0, 0]):
    #     self.car.speed = 0.0
    keys_held = pygame.key.get_pressed()
    acceleration = 0.0
    steering = 0.0
    if keys_held[pygame.K_LEFT]:
        steering -= 1.0
    if keys_held[pygame.K_RIGHT]:
        steering += 1.0
    if keys_held[pygame.K_UP]:
        acceleration += 1.0
    if keys_held[pygame.K_DOWN]:
        acceleration -= 1.0
    return acceleration, steering


if __name__ == '__main__':
    game = Game()
    game.run()
