import minihack
import nle.nethack.actions as actions
import minihack.dat
import time
from utility import print_inventory

from AgentMinihack import AgentNetHack



# Percorso al tuo file .des

des_file = "level/closed_door.des"
des_file = "level/monster_f.des"
des_file = "level/potion.des"
#des_file = "level/weapon.des"

files = []
files.append("level/closed_door.des")
files.append("level/potion.des")
files.append("level/weapon.des")



#with open(files[0], "r") as f:
#        des_content = f.read()
#env = minihack.MiniHack(des_file=des_content, actions=actions.ACTIONS, max_episode_steps=200, pet=False, observation_keys=['glyphs', 'tty_chars', 'tty_colors', 'chars', 'colors', 'specials', 'glyphs_crop', 'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message','inv_strs', 'inv_letters'])
#obs, info = env.reset()
#start = time.time()
#a = AgentNetHack(env)
#a.move()
#a.save()
#print(f"turni {a.turni}")
#print(f"time: {time.time()-start}")


##train
files = ["level/monster_f.des"]
files.append("level/monster_d.des")
files.append("level/monster_o.des")
files.append("level/monster_O.des")
files.append("level/monster_fo.des")

game = []


for des_file in files:
    # Leggi il contenuto del file .des
    with open(des_file, "r") as f:
        des_content = f.read()
    for i in range(2):
        env = minihack.MiniHack(des_file=des_content, actions=actions.ACTIONS, max_episode_steps=200, pet=False, observation_keys=['glyphs', 'tty_chars', 'tty_colors', 'chars', 'colors', 'specials', 'glyphs_crop', 'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message','inv_strs', 'inv_letters'])
        obs, info = env.reset()
        start = time.time()
        a = AgentNetHack(env)
        win = a.move()
        a.save()
        print(f"turni {a.turni}")
        print(f"time: {time.time()-start}")
        game.append((win, a.turni, des_file))

print(game)

import pandas as pd

# Crea DataFrame
df = pd.DataFrame(game, columns=['win', 'turns', 'file'])

# Raggruppa per file, calcola n vittorie e media turni
results = df.groupby('file').agg(
    vinte=('win', 'sum'),
    partite=('win', 'count'),
    turni_medi=('turns', 'mean')
).reset_index()

# Ordina per file
results = results.sort_values('file')

print(results)



