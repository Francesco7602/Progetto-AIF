import minihack
import nle.nethack.actions as actions
import minihack.dat

from Agent import AgentNetHack
from utility import ascii_to_idx, print_stats, print_inventory, Simboli_unici
import re

level = minihack.LevelGenerator()

# Percorso al tuo file .des
des_file = "dat/chest.des"

# Leggi il contenuto del file .des
with open(des_file, "r") as f:
    des_content = f.read()

env = minihack.MiniHack(des_file=des_content, actions=actions.ACTIONS, max_episode_steps=200, pet=False, observation_keys=['glyphs', 'tty_chars', 'tty_colors', 'chars', 'colors', 'specials', 'glyphs_crop', 'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message','inv_strs', 'inv_letters'])

obs, info = env.reset()


#Simboli_unici(obs)



a = AgentNetHack(env)
a.move()
a.save()
print(f"turni {a.turni}")

#env.render()
#
#print_inventory(obs)
#terminal = truncated = False
#while not (terminal or truncated):
#    print_stats(obs)
#    env.render()
#    cmd = ascii_to_idx(input("cmd: "))  
#    obs, reward, terminal, truncated, info = env.step(cmd)
#
#print(f"{truncated}, {terminal}")
