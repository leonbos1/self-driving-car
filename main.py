import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import math
from model import DQN
import random

IMAGES_PATH = "./images/"
MAP_SIZE = (800, 600)

RIGHT = 'RIGHT'
LEFT = 'LEFT'
FRONT = 'FRONT'
LEFT_FRONT = 'LEFT_FRONT'
RIGHT_FRONT = 'RIGHT_FRONT'


class Road:
    def __init__(self):
        self.image = pygame.image.load(IMAGES_PATH + "road.png")
        self.image = pygame.transform.scale(self.image, MAP_SIZE)

    def draw(self, screen):
        screen.blit(self.image, (0, 0))


class Car:
    def __init__(self):
        self.original_image = pygame.image.load(IMAGES_PATH + "car.png")
        self.image = self.original_image
        self.x = 350
        self.y = 450
        self.angle = 0
        self.speed = 3

        self.rect = self.image.get_rect(center=(self.x, self.y))

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def move(self):
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y -= self.speed * math.sin(math.radians(self.angle))

        self.rect = self.image.get_rect(center=(self.x, self.y))

    def rotate(self, direction, steering_angle=5):
        if direction == RIGHT:
            if self.angle - steering_angle < 0:
                self.angle = 355
            else:
                self.angle -= steering_angle

            self.image = pygame.transform.rotate(
                self.original_image, self.angle)

        elif direction == LEFT:
            if self.angle + steering_angle >= 360:
                self.angle = 0
            else:
                self.angle += steering_angle

            self.image = pygame.transform.rotate(
                self.original_image, self.angle)

    # def rotate(self, angle):
    #     angle = angle % 360

    #     self.angle = angle
    #     self.image = pygame.transform.rotate(self.original_image, self.angle)

    def is_colliding(self, road):
        return road.image.get_at((int(self.x), int(self.y))) == (255, 255, 255, 255)

    def get_wall_distance(self, road, direction):
        if direction == RIGHT:
            angle = self.angle - 90
        elif direction == LEFT:
            angle = self.angle + 90
        elif direction == FRONT:
            angle = self.angle
        elif direction == LEFT_FRONT:
            angle = self.angle + 45
        elif direction == RIGHT_FRONT:
            angle = self.angle - 45

        x = self.x
        y = self.y

        while road.image.get_at((int(x), int(y))) != (255, 255, 255, 255):
            x += math.cos(math.radians(angle))
            y -= math.sin(math.radians(angle))

        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def get_state(self, road):
        return torch.tensor([[
            self.get_wall_distance(road, LEFT),
            self.get_wall_distance(road, RIGHT),
            self.get_wall_distance(road, FRONT),
            self.get_wall_distance(road, LEFT_FRONT),
            self.get_wall_distance(road, RIGHT_FRONT),
            self.angle / 360
        ]])


class Game:
    def __init__(self):
        self.running = True
        self.screen = pygame.display.set_mode(MAP_SIZE)
        self.clock = pygame.time.Clock()
        self.car = Car()
        self.road = Road()
        self.model = DQN(6, 128, 2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return torch.tensor([[random.randrange(2)]], dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return q_values.max(1)[1].view(1, 1)

    def run(self):
        exit = False
        epsilon_start = 0.9
        epsilon_end = 0.05
        epsilon_decay = 200
        steps_done = 0

        for episode in range(1000):
            if exit:
                break

            state = self.car.get_state(self.road)
            total_reward = 0

            self.car = Car()

            while not self.car.is_colliding(self.road) and self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        exit = True

                epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                    math.exp(-1. * steps_done / epsilon_decay)

                action = self.select_action(state, epsilon)
                steps_done += 1

                reward, next_state = self.step(action)
                total_reward += reward

                expected_state_action_values = reward + \
                    self.gamma * self.model(next_state).max(1)[0]
                
                state_action_values = self.model(state)[0, action]

                loss = self.criterion(
                    state_action_values, expected_state_action_values)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state

                collision = self.car.is_colliding(self.road)

                # if collision:
                #     self.restart()

                self.screen.fill((0, 0, 0))
                self.road.draw(self.screen)
                self.car.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(300)

            print(f"Episode: {episode}, Total Reward: {total_reward}")

    def step(self, action):
        if action == 0:
            self.car.rotate(LEFT)
        elif action == 1:
            self.car.rotate(RIGHT)

        self.car.move()

        if self.car.is_colliding(self.road):
            reward = -100
            next_state = self.car.get_state(self.road)

        else:
            reward = 1

        next_state = self.car.get_state(self.road)

        return reward, next_state

    def restart(self):
        self.car = Car()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
