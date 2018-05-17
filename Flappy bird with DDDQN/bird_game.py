# Flappy bird - AI version
import numpy as np
import time

class env_train():

    def __init__(self):
        import pygame
        import time
        import random

        # Initialize pygame
        #pygame.init()

        # Window size (unit = pixel)
        self.display_width = 700
        self.display_height = 700

        # RGB
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.block_color = (53, 115, 255)

        # Car image's width (To calculate if it avoids the obstacle)
        self.bird_height = 58

        # Set a window
        #self.gameDisplay = pygame.display.set_mode( (display_width, display_height) )

        # Set a title of the window
        #pygame.display.set_caption('Flapping Bird')

        # Set speed of the game preceeding at the end of the loop with it
        self.clock = pygame.time.Clock()

        # Import a car image
        #self.carImg = pygame.image.load('C:\\Users\\Administrator\\Desktop\\bird1_m.png')
        #self.carImg2 = pygame.image.load('C:\\Users\\Administrator\\Desktop\\bird2_m.png')  # wing_up
        #self.forest = pygame.image.load('C:\\Users\\Administrator\\Desktop\\forest.png')


        ##############  Key information for the agent   ###################
        # Car's coordinate
        self.x = 0  # initialize
        self.y = 0

        # Obstacle's
        self.thing_startx = 0
        self.thing_starty = 0
        self.thing_speed = 0
        self.thing_width = 0
        self.thing_height = 0

        # Reward
        self.reward = 0  # initialize

        # Done
        self.done = False  # initialize

        # next_state
        self.next_state = 0  # initialize

        self.iter = 0 # needed to represent flapping motion

        self.t = 0 # needed to represent gravity force
        #################################

        self.flap = False

    def things_dodged(self, count):
        import pygame
        font = pygame.font.SysFont(None, 25)
        text = font.render("Reward: " + str(count), True, self.black)
        self.gameDisplay.blit(text, (0, 0))

    def things(self, thingx, thingy, thingw, thingh, color):
        import pygame
        pygame.draw.rect(self.gameDisplay, color, [thingx, thingy, thingw, thingh])

    def car(self, img, x, y):
        self.gameDisplay.blit( img, (x, y))

    def text_objects(self, text, font):
        textSurface = font.render(text, True, self.black)
        return textSurface, textSurface.get_rect()

    def message_display(self, text):  # for a message, when the car crashes.
        import pygame
        largeText = pygame.font.Font('freesansbold.ttf', 115)
        TextSurf, TextRect = self.text_objects(text, largeText)
        TextRect.center = ((self.display_width / 2), (self.display_height / 2))
        self.gameDisplay.blit(TextSurf, TextRect)

        pygame.display.update()

    # "RESET" THE ENVIRONMENT!
    # When you restart your game after crashed, you use this function in agent.py.
    def reset(self):
        import random

        # We will restart the same with this function.
        # In agent.py, "state = env.reset()" will be used.

        # Car's coordinate
        self.x = (self.display_width * 0.04) # == acc_x
        self.y = (self.display_height * 0.5) # == acc_y

        # Obstacle's coordinate

        self.thing_startx = self.display_width
        self.thing_starty = random.randrange(0 + self.bird_height * 2, self.display_height - self.bird_height * 2)
        self.thing_speed = 4
        self.thing_width = 85
        self.thing_height = 700

        # Reward
        self.reward = 0

        # Done
        self.done = False

        self.t = 0  # needed to represent gravity force

        self.flap = False

        return [self.x, self.y , self.t , self.thing_startx , self.thing_starty ]

    def crash(self, ):
        self.message_display('You Crashed')

    def step(self, action):  # "step( action )" Part
        '''
        NOTE : If you'd like to change the state, change that in reset(), step() both.
        '''
        import pygame
        import random

        #########################################################################################
        # Move the agent
        encoded_action = action
        action_decoder = {0: -13*1 , 1:0}  # 0:flapping , 1:doing nothing

        # Color the background
        #self.car(self.forest, 0, 0)

        if encoded_action == 0 :
            self.flap = True
            self.y += action_decoder[encoded_action]

            self.t = 0
            #self.car(self.carImg, self.x, self.y)

        if encoded_action == 1 :
            self.flap = False
            self.y += 1 + self.t**2 # gravity force
            self.t += 0.1  #

            #if self.iter%10==0:
                #self.car(self.carImg, self.x, self.y)
            #else:
                #self.car(self.carImg2, self.x, self.y)

        self.flap = False
        self.iter += 1

        #print('self.accumulated_acc :', self.accumulated_acc)

        # Draw the obstacles
        # Lower obstacle
        #self.things(self.thing_startx, self.thing_starty, self.thing_width, self.thing_height, self.block_color)
        # Upper obstacle
        #self.things(self.thing_startx, self.thing_starty - self.thing_height - 2 * self.bird_height, self.thing_width, self.thing_height, self.block_color)


        # Move the obstacle
        self.thing_startx -= self.thing_speed

        # Draw a score display
        #self.things_dodged(self.reward)

        # Reward at each step
        reward_at_this_step = 0


        # Wall crash condition
        if self.y < 0 or self.y > self.display_height - self.bird_height:
            #self.crash()
            reward_at_this_step = -20

            self.reward = self.reward - 20

            self.done = True

        # Obstacle crash condition
        if self.x >= self.thing_startx - self.thing_width:
            if self.y < self.thing_starty - 2 *self. bird_height or self.y +self. bird_height > self.thing_starty:
                #self.crash()

                self.reward = self.reward = -20
                reward_at_this_step = -20

                self.done = True

        # Re-generate the obstacle
        if self.thing_startx + self.thing_width < 0:
            self.thing_startx = self.display_width
            self.thing_starty = random.randrange(0 + self.bird_height * 2, self.display_height - self.bird_height * 2)

            # Reward
            self.reward += 1
            reward_at_this_step = 1


        # Define next_state
        self.next_state = [self.x, self.y , self.t , self.thing_startx , self.thing_starty ]

        #pygame.display.update()
        #self.clock.tick(60) # Based on FPS (Frame Per Second)

        return [self.next_state, reward_at_this_step , self.done ] # self.rewarad = total_reward

    def total_reward_clear(self):
        self.reward = 0

    def reset_terminal(self):
        self.done = False









class env_replay():

    def __init__(self):
        import pygame
        import time
        import random

        # Initialize pygametotal_reward_clear
        pygame.init()

        # Window size (unit = pixel)
        self.display_width = 700
        self.display_height = 700

        # RGB
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.block_color = (53, 115, 255)

        # Car image's width (To calculate if it avoids the obstacle)
        self.bird_height = 58

        # Set a window
        self.gameDisplay = pygame.display.set_mode( (self.display_width, self.display_height) )

        # Set a title of the window
        pygame.display.set_caption('Flapping Bird')

        # Set speed of the game preceeding at the end of the loop with it
        self.clock = pygame.time.Clock()

        # Import a car image
        self.carImg = pygame.image.load('bird1_m.png')
        self.carImg2 = pygame.image.load('bird2_m.png')  # wing_up
        self.forest = pygame.image.load('forest.png')

        ##############  Key information for the agent   ###################
        # Car's coordinate
        self.x = 0  # initialize
        self.y = 0

        # Obstacle's
        self.thing_startx = 0
        self.thing_starty = 0
        self.thing_speed = 0
        self.thing_width = 0
        self.thing_height = 0

        # Reward
        self.reward = 0  # initialize

        # Done
        self.done = False  # initialize

        # next_state
        self.next_state = 0  # initialize

        self.iter = 0 # needed to represent flapping motion

        self.t = 0 # needed to represent gravity force
        #################################

        self.flap = False

    def things_dodged(self, count):
        import pygame
        font = pygame.font.SysFont(None, 25)
        text = font.render("Reward: " + str(count), True, self.black)
        self.gameDisplay.blit(text, (0, 0))

    def things(self, thingx, thingy, thingw, thingh, color):
        import pygame
        pygame.draw.rect(self.gameDisplay, color, [thingx, thingy, thingw, thingh])

    def car(self, img, x, y):
        self.gameDisplay.blit( img, (x, y))

    def text_objects(self, text, font):
        textSurface = font.render(text, True, self.black)
        return textSurface, textSurface.get_rect()

    def message_display(self, text):  # for a message, when the car crashes.
        import pygame
        largeText = pygame.font.Font('freesansbold.ttf', 115)
        TextSurf, TextRect = self.text_objects(text, largeText)
        TextRect.center = ((self.display_width / 2), (self.display_height / 2))
        self.gameDisplay.blit(TextSurf, TextRect)

        pygame.display.update()

    # "RESET" THE ENVIRONMENT!
    # When you restart your game after crashed, you use this function in agent.py.
    def reset(self):
        import random

        # We will restart the same with this function.
        # In agent.py, "state = env.reset()" will be used.

        # Car's coordinate
        self.x = (self.display_width * 0.04) # == acc_x
        self.y = (self.display_height * 0.5) # == acc_y

        # Obstacle's coordinate

        self.thing_startx = self.display_width
        self.thing_starty = random.randrange(0 + self.bird_height * 2, self.display_height - self.bird_height * 2)
        self.thing_speed = 4
        self.thing_width = 85
        self.thing_height = 700

        # Reward
        self.reward = 0

        # Done
        self.done = False

        self.t = 0  # needed to represent gravity force

        self.flap = False

        return [self.x, self.y , self.t , self.thing_startx , self.thing_starty ]

    def crash(self, ):
        self.message_display('You Crashed')

    def step(self, action):  # "step( action )" Part
        '''
        NOTE : If you'd like to change the state, change that in reset(), step() both.
        '''
        import pygame
        import random

        #########################################################################################
        # Move the agent
        encoded_action = action
        action_decoder = {0: -13*1 , 1:0}  # 0:flapping , 1:doing nothing

        # Color the background
        self.car( self.forest, 0, 0)

        if encoded_action == 0 :
            self.flap = True
            self.y += action_decoder[encoded_action]

            self.t = 0
            self.car(self.carImg, self.x, self.y)

        if encoded_action == 1 :
            self.flap = False
            self.y += 1 + self.t**2 # gravity force
            self.t += 0.1  #

            if self.iter%7==0:
                self.car(self.carImg, self.x, self.y)
            else:
                self.car(self.carImg2, self.x, self.y)

        self.flap = False
        self.iter += 1

        # Draw the obstacles
        # Lower obstacle
        self.things(self.thing_startx, self.thing_starty, self.thing_width, self.thing_height, self.block_color)
        # Upper obstacle
        self.things(self.thing_startx, self.thing_starty - self.thing_height - 2 * self.bird_height, self.thing_width, self.thing_height, self.block_color)


        # Move the obstacle
        self.thing_startx -= self.thing_speed

        # Draw a score display
        self.things_dodged(self.reward)

        # Reward at each step
        reward_at_this_step = 0


        # Wall crash condition
        if self.y < 0 or self.y > self.display_height - self.bird_height:
            reward_at_this_step = -20

            self.reward = self.reward - 20

            self.done = True

        # Obstacle crash condition
        if self.x >= self.thing_startx - self.thing_width: # self.x >= self.thing_startx-self.thing_width
            if self.y < self.thing_starty - 2 *self. bird_height or self.y +self. bird_height > self.thing_starty:

                self.reward = self.reward = -20
                reward_at_this_step = -20

                self.done = True

        # Re-generate the obstacle
        if self.thing_startx + self.thing_width < 0:
            self.thing_startx = self.display_width
            self.thing_starty = random.randrange(0 + self.bird_height * 2, self.display_height - self.bird_height * 2)

            # Reward
            self.reward += 1
            reward_at_this_step = 1


        # Define next_state
        self.next_state = [self.x, self.y , self.t , self.thing_startx , self.thing_starty ]

        pygame.display.update()
        # self.clock.tick(60) # Based on FPS (Frame Per Second)
        time.sleep(0.03)

        return [self.next_state, reward_at_this_step , self.done ] # self.rewarad = total_reward

    def total_reward_clear(self):
        self.reward = 0

    def reset_terminal(self):
        self.done = False




# 'EXECUTE
import pygame

'''
import time

env = env()

# Reset
env.reset()

# Proceed
env.step( 2 )
time.sleep(2)

# Proceed
env.step(2)
time.sleep(2)

# Proceed
env.step(2)
time.sleep(2)

# The game window will be closed whehter or not the codes below exist.
pygame.quit()
quit()
'''


'''
NOTE : YOU CAN VISUALIZE DP OR WHATEVER SYSTEM WITH PYGAME!!

- As long as game_loop()s are followed by another game_loop()s, the window is not closed. ( It's not even closed while waiting for DQN to be calculated. )
- Make this "class" format.
e.g.
game_loop() # 1st scene
..
state = env.reset()
..
next_state, reward, done = env.step(action)
..
game_loop() # Updated scene from the 1st
'''
