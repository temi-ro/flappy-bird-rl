import pygame
import random
import neat
import pickle
import os
import argparse
from helper import plot

# Size of the window
WIDTH = 700
HEIGHT = 900

BLOCK_WIDTH = 100
BLOCK_HEIGHT = 700
BLOCK_HEIGHT_MIN = HEIGHT//3
GAP_SIZE = 300
PLAYER_SIZE = 50

SPEED = 200 # Usually 40 but 200 for training
MOVE = 10
STEP_JUMP = 65 # Player's step up
GRAVITY = 6 # Player's step down
START_POS_X = 100
HEIGHT = 1080

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

pygame.init()
font = pygame.font.Font('arial.ttf',    40)
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")


class Bird():
    def __init__(self) -> None:
        self.rect = pygame.Rect(START_POS_X, HEIGHT/2, PLAYER_SIZE, PLAYER_SIZE)
        self.y = self.rect.top
        self.x = self.rect.left
        if args.color == "red":
            self.color = RED
        else:
            self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def move(self):
        self.rect.y += GRAVITY
        self.y += GRAVITY

    def jump(self):
        self.rect.y -= STEP_JUMP
        self.y -= STEP_JUMP

    def draw(self):
        pygame.draw.rect(WINDOW, self.color, self.rect)

class Pipe(): 

    def __init__(self, x):
        rand_height = random.randint(BLOCK_HEIGHT_MIN, HEIGHT - BLOCK_HEIGHT_MIN)
        self.up = pygame.Rect(x, 0, BLOCK_WIDTH, rand_height)
        self.down = pygame.Rect(x, rand_height + GAP_SIZE, BLOCK_WIDTH, HEIGHT)
        self.height = rand_height

    def move(self):
        self.up.x -= MOVE
        self.down.x -= MOVE
    
    def draw(self):
        pygame.draw.rect(WINDOW, WHITE, self.up)
        pygame.draw.rect(WINDOW, WHITE, self.down)

class FlappyBird:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()

        self.pipe = None
        self.reset()

    def reset(self):
        self.frame_iter = 0
        self.score = 0
    
    
    def is_collision(self, player, pipe):
        if player.y <= 0 or player.y + PLAYER_SIZE > self.h:
            return True
        
        hitbox_x = player.x + PLAYER_SIZE
        

        if (hitbox_x >= pipe.up.left and hitbox_x <= pipe.up.left + BLOCK_WIDTH) and (player.y <= pipe.up.bottom or player.y + PLAYER_SIZE >= pipe.down.top):
            return True

        return False

    def update_ui(self, score):
        # Score txt
        text = font.render("Score: " + str(score), True, WHITE)
        self.display.blit(text, [0, 0])


def eval_gen(genomes, config, stop=True):
    game = FlappyBird()
    pipe = Pipe(WIDTH)
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        ge.append(genome)

    run = True
    clock = pygame.time.Clock()
    score = 0
    while run and len(birds) > 0:
        clock.tick(SPEED)
        WINDOW.fill(BLACK)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                quit()

        for x, bird in enumerate(birds):
            # Rewards bird for being alive
            ge[x].fitness += 0.1
            bird.move()       
            bird.draw()
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipe.up.bottom), abs(bird.y - pipe.down.top), abs(bird.x - pipe.up.left)))
            if output[0] > 0.5:
                bird.jump()

            # If bird collides with pipe, then kill it and show it is wrong
            if game.is_collision(bird, pipe):
                ge[birds.index(bird)].fitness -= 1
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

            # If score big enough, end generation and reward all birds alive
            if stop and score > 50:
                ge[birds.index(bird)].fitness += 20
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))
            
        pipe.move()
        pipe.draw()

        # Delete pipe and add new one
        if pipe == None or pipe.up.x <= -BLOCK_WIDTH:
            del pipe
            score += 1
            pipe = Pipe(WIDTH+BLOCK_WIDTH)

        game.update_ui(score)
        pygame.display.flip()


def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    if args.checkpoint:
        p = neat.Checkpointer.restore_checkpoint(f"checkpoints/neat-checkpoint-{args.checkpoint}")
    p.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=50, filename_prefix="checkpoints/neat-checkpoint-"))

    winner = p.run(eval_gen, args.nGens) # run eval_gen for nGens generations (default=100)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def replay_genome(config_path, genome_path="best.pickle"):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    eval_gen(genomes, config, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flappy Bird AI')
    parser.add_argument('--config', type=str, help='Path to the NEAT config file', default='config-ff.txt')
    parser.add_argument("--color", type=str, help="Color of the bird (random or red)", default="random")
    parser.add_argument("--nGens", type=int, help="Number of generations to run", default=100)
    parser.add_argument("--checkpoint", type=str, help="Checkpoint number to restore", default=None)
    parser.add_argument("--replay", type=str, help="Path to the genome to replay", default=None)
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, args.config)

    if args.replay:
        replay_genome(config_path, args.replay)
    else:
        run_neat(config_path)

     
