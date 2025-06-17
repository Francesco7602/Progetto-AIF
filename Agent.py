from pyswip import Prolog
from pathlib import Path
from utility import a_star, Simboli_unici, SymbolToPos, save, load, ascii_to_idx
import os
import time

class AgentNetHack:
    def load(self):
        if Path("memory").exists():
            load("memory", self.prolog)
    def save(self):
        arr=[]
        arr.append("risk_zone_symbol(X,Y)")
        arr.append("walkable(X,Y)")
        arr.append("openable(X,Y)")
        arr.append("command(X,Y,Z)")
        arr.append("is_known(X,Y,Z)")
        arr.append("is_useful(X,Y,V,Z)")
        arr.append("is_monster(X,Y)")
        with open("memory", "w") as f:
            for e in arr:
                str = e.split("(")
                f.write(save(e,str[0],self.prolog))
        

    def __init__(self, env):
        self.env=env
        self.obs, self.info = env.reset()

        blstats = self.obs["blstats"]
        self.pos = (blstats[0].item(), blstats[1].item())  # x, y

        self.strength = blstats[3]
        self.dexterity = blstats[4]
        self.constitution = blstats[5]
        self.intelligence = blstats[6]
        self.wisdom = blstats[7]
        self.charisma = blstats[8]
        self.score = blstats[9]
        self.hp = (blstats[10], blstats[11])  # current, max
        self.energy = (blstats[14], blstats[15])  # current, max
        self.ac = blstats[16]

        self.depth = blstats[12]
        self.gold = blstats[13]
        
        self.monster_level = blstats[17]
        self.experience_level = blstats[18]
        self.experience_points = blstats[19]
        self.time = blstats[20]
        self.hunger_state = blstats[21]
        self.carrying_capacity = blstats[22]
        self.dungeon_number = blstats[23]
        self.level_number = blstats[24]

        msg = "".join([chr(c) for c in self.obs["message"]]).strip()
        inv_letters = [chr(c) for c in self.obs["inv_letters"] if c != 0]

        self.height = len(self.obs["tty_chars"])
        self.width = len(self.obs["tty_chars"][0])
        self.prolog=Prolog()
        self.prolog.consult("kb.pl")
        self.unknow = set()
        self.explored = {}
        self.goals = None

    def goal(self):
        arr = Simboli_unici()



    #    
    #    if cur_goal:
    #        return cur_goal
    #    tty_chars = obs["tty_chars"]
    #    tty_colors = obs["tty_colors"]
#
    #    goal = set()
    #    map = set()
#
    #    height = len(tty_chars)
    #    width = len(tty_chars[0])
#
    #    for y in range(height):
    #        for x in range(width):
    #            code = tty_chars[y][x]
    #            char = chr(tty_chars[y][x])
    #            color = tty_colors[y][x]
    #            #if color == 7 and char in '.':
    #            #if char not in ['|', '-'] or (char=='+' and color == 3):
    #            
    #            #print (f"search in KB {code} {color}")
    #            results=list(self.prolog.query(f"walkable(({code},{color}), X)"))
    #            if len(results)==0 and (code, color) not in self.unknow:
    #                self.unknow.add((code, color))
    #                print (f"Nuovo oggetto da scoprire {char} {code} {color}")
    #            elif ((code, color) not in self.unknow) and results[0]['X']=='true':
    #                map.add((y, x))
    #            
    #            if color == 3 and (char == '+' or char == '-' or char == '|'):
    #                print(f"questa è una porta: {char}, {color}")
    #                goal.add((x,y-1, '+'))
#
    #    for y, x in map:
    #        #if chr(tty_chars[y][x]) == '.':
#
    #        vicini = [(y+dy, x+dx) for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]]
    #        count = sum((ny, nx) in map and chr(tty_chars[ny][nx]) not in ['|', '-'] for ny, nx in vicini)
    #        if count == 1:
    #            goal.add((x,y-1, '.'))
    #        if count == 0:
    #            vicini = [(y+dy, x+dx) for dy, dx in [(1,1), (1,-1), (-1,1), (-1,-1)]]
    #            count = sum((ny, nx) in map and chr(tty_chars[ny][nx]) not in ['|', '-'] for ny, nx in vicini)
    #            if count == 1:
    #                goal.add((x,y-1, '.'))           
#
    #    print("------ Goal ------")
    #    for g in sorted(goal):
    #        print(g)
#
    #    return goal.pop()
    #
    #
    def move_to(self, pos_cur, pos_prox):
        x, y = pos_cur
        x_goal, y_goal = pos_prox
        direction = (1 if x > x_goal else (0 if x == x_goal else -1) , 1 if y > y_goal else (0 if y == y_goal else -1))
        match direction:
            case (1,0):
                return 3 #move to W
            case (-1,0):
                return 1 #move to E
            case (0,1):
                return 0 #move to N
            case (0,-1):
                return 2 #move to S
            
            case (1,1):
                return 7 #move to NW
            case (1,-1):
                return 6 #move to SW
            case (-1,1):
                return 4 #move to NE
            case (-1,-1):
                return 5 #move to SE
            case (0,0):
                return None
    
    def heuristic(self, pos, goal, env):
        """used Chebyshev"""
        x, y = pos
        x_goal, y_goal = goal

        h = max(abs(x - x_goal), abs(y - y_goal)) + self.logical_penalty(pos, goal, env)
        #h = abs(x - x_goal)+ abs(y - y_goal) + self.logical_penalty(pos, env)
        return h
    
    def logical_penalty(self, pos, goal, env):
        term = 0
        x, y = pos
        code = env["tty_chars"][y][x]
        color = env["tty_colors"][y][x]
        results=list(self.prolog.query(f"risk_zone_symbol(({code},{color}), X)"))
        
        if len(results)>0:
            term = int(results[0]['X'])
        results=list(self.prolog.query(f"risk_zone_pos(({x},{y}), X)"))
        
        if len(results)>0:
            term += int(results[0]['X'])

        
        return term


    def is_walkable(self, pos, env):
        x, y = pos
        code = env["tty_chars"][y][x]
        color = env["tty_colors"][y][x]
          

        results=list(self.prolog.query(f"walkable(({code},{color}), X)"))
        
        if len(results)==0:
            if (code, color) not in self.unknow:
                self.unknow.add((code, color))
                print(f"fun is_walkable: New object to discover {chr(code)} {code} {color}")
            return True  # initially consider it walkable
        else:
            return results[0]["X"] == 'true'
        

    def neighbors(self, x, y, env):
        result = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            results = list(self.prolog.query(f"moveInvalid(({x},{y}), ({nx},{ny}), X)"))
            if len(results)>0:
                continue
            
            if 0 <= nx < self.width and 0 <= ny < self.height:
                
                if self.is_walkable((nx, ny), env):
                    result.append((nx, ny))
        return result

    #def move(self):
    #    
    #    x, y, ob = self.goal(self.obs, cur_goal = None)
    #    curr = (x, y, ob)
    #    print (f"fun move: goal: {self.pos} -> {curr}")
    #    goal = (x,y+1)
    #    xp, yp = self.pos
    #    start = (xp, yp+1)
    #    print(f"start posizione mappa {self.pos} start")
    #    path = a_star(start, goal, self, self.obs)
    #    print (f"fun move: goal path: {path}")
    #    
#
    #    if path is None:
    #        print("No path found.")
    #        return False  # movimento fallito
#
    #    for step in path[1:]:  # salta il primo (start)
    #        if step == goal:
    #            cmd
    #        cmd = self.move_to(start, step)  # esegui lo spostamento logico/fisico
    #        print(f"Sono in {start} e vado in {step} e faccio cmd:{cmd}")
    #        obs, reward, done, truncated, info = self.env.step(cmd)
    #        render(obs)
    #        if self.observe_and_update(step, obs, cmd):  # aggiorna KB con nuove info
    #            print("nuovo goal")
    #            self.move()
    #            print("fine nuovo goal")
    #            return
    #        xp, yp = self.pos
    #        start = (xp, yp+1)
#
#
    #        #if self.check_environment_change(step, goal, obs):
    #        #    print("Environment changed, recalculating path...")
    #        #    return self.move(step, self.select_new_goal(env), env)  # ricorsione
    #    print("goal raggiunto")
    #    return True  # goal raggiunto


    def move(self, goal=None):
        start = (self.pos[0], self.pos[1]+1)
        #os.system('clear')  # Pulisce il terminale
        self.env.render()
        #time.sleep(0.5)  # Aspetta un po' per vedere il frame
        self.goals = SymbolToPos(self.obs, self.prolog, self.explored, self.goals)

        #print(goals)
        path = None
        if self.goals[0][2] >= 6:
            print("Hahaha mostro")

        else:

            goal = self.goals[0][1]
            path = a_star(start, goal, self, self.obs)
            print (f"fun move: goal path: {path}")
        obj={}
        if path is None:
            return
        for step in path[1:]:
            start = (self.pos[0], self.pos[1]+1)
            self.explored[step]=1
            self.pos = (self.obs["blstats"][0].item(), self.obs["blstats"][1].item())
            obj['code']=self.obs['tty_chars'][step[1]][step[0]]
            obj['color']=self.obs['tty_colors'][step[1]][step[0]]

            results = list(self.prolog.query(f"is_known(Y,({obj['code']},{obj['color']}), X)"))
            cmdlist = [ascii_to_idx('o'), ascii_to_idx('c')]
            print(f"Debug sono (start {start}) ({self.pos[0]},{self.pos[1]+1}) vado in ({step}) symbol: {chr(obj['code'])} ({obj['code']}, {obj['color']}), results {results}")
            if type(cmdlist) != list:
                cmdlist = [cmdlist]

            if len(results) == len(cmdlist):
                tmpcmd = cmdlist
                for i in results:
                    if i.get('X')== 'false' and i.get('Y') in tmpcmd:
                        tmpcmd.remove(i.get('Y'))
            else:
                tmpcmd=cmdlist
                for i in results:
                    if i.get('Y') in tmpcmd:
                        tmpcmd.remove(i.get('Y'))

            print("AZIONE APPLICATA A :")
            print(tmpcmd, obj['code'], obj['color'])
            #if len(results)==0 or (len(results)>0 and results[0].get('X', 'true') == 'true'):
            if step == goal:

                print("sono accanto al goal")
                for cmd in tmpcmd:
                    self.obs, reward, terminal, truncated, info = self.env.step(cmd)
                    #os.system('clear')  # Pulisce il terminale
                    self.env.render()
                    #time.sleep(0.5)  # Aspetta un po' per vedere il frame
                    cmd1 = self.move_to(start, step)  # direction
                    self.obs, reward, terminal, truncated, info = self.env.step(cmd1)
                    #os.system('clear')  # Pulisce il terminale
                    self.env.render()
                    #time.sleep(0.5)  # Aspetta un po' per vedere il frame
            else:
                cmd = self.move_to(start, step)


        else:
            cmd = self.move_to(start, step)  # esegui lo spostamento logico/fisico
        self.obs, reward, terminal, truncated, info = self.env.step(cmd)
        if terminal or truncated:
            print(f"bravo {reward}")
            return
        #os.system('clear')  # Pulisce il terminale
        self.env.render()
        #time.sleep(0.5)  # Aspetta un po' per vedere il frame
        if self.observe_and_update(step, cmd=cmd, obj=obj):  # aggiorna KB con nuove info
            print("nuovo goal")
            self.move()
            return
        self.move()
        return
        print (f"search a goal")
        if not goal:
            x, y, ob = self.goal(self.obs, cur_goal = None)
            goal = (x,y+1)
        start = (self.pos[0], self.pos[1]+1)
        print (f"fun move: goal: {self.start} -> {goal}")
        #print(f"start posizione mappa {self.pos} start")
        path = a_star(start, goal, self)
        print (f"fun move: goal path: {path}")

        if path is None:
            print("No path found.")
            return False  # movimento fallito

        for step in path[1:]:
            if step == goal:
                print("sono accanto al goal")
            else:
                cmd = self.move_to(start, step)  # esegui lo spostamento logico/fisico
                obs, reward, terminal, truncated, info = self.env.step(cmd)
                #os.system('clear')  # Pulisce il terminale
                self.env.render()
                #time.sleep(0.5)  # Aspetta un po' per vedere il frame
            if self.observe_and_update(step, obs, cmd):  # aggiorna KB con nuove info
                print("nuovo goal")
                self.move()
                print("fine nuovo goal")
                return
            xp, yp = self.pos
            start = (xp, yp+1)


            #if self.check_environment_change(step, goal, obs):
            #    print("Environment changed, recalculating path...")
            #    return self.move(step, self.select_new_goal(env), env)  # ricorsione
        print("goal raggiunto")
        return True  # goal raggiunto


    def observe_and_update(self, step, cmd=None, obj=None):
        blstats = self.obs["blstats"]
        print(f"cmd {cmd}, mappa pos: {self.pos}")

        code = obj['code']
        color = obj['color']
        if cmd<8:
            cont = 0
            if (self.pos == (blstats[0].item(), blstats[1].item())):
                self.prolog.assertz(f"moveInvalid(({self.pos[0]},{self.pos[1]+1}), ({step[0]},{step[1]}), 10)")
                print(f"insert moveInvalid {self.pos[0]}, {self.pos[1]+1} to ({step[0]},{step[1]})")
                self.explored[step]=0
                return True
            results = list(self.prolog.query(f"is_known(0,({code},{color}), X)"))
            if len(results)==0:
                results = list(self.prolog.query(f"command(0,({code},{color}), X)"))
                if (self.pos == (blstats[0].item(), blstats[1].item())):
                    print(f"probabile muro {step} element {code} {color}")
                    self.prolog.assertz(f"risk_zone_pos(({step[0]},{step[1]}),10)")
                    if len(results)==0:
                        self.prolog.assertz(f"command(0,({code},{color}), 1)")
                    else:
                        
                        list(self.prolog.query(f"retractall(command(0,({code},{color}), _))"))
                        cont = (results[0]['X'])
                        self.prolog.assertz(f"command(0,({code},{color}), {cont+1})")
                    if cont>5:
                        self.prolog.assertz(f"walkable(({code},{color}), false)")
                        self.prolog.assertz(f"is_known(0,({code},{color}), true)")
                        list(self.prolog.query(f"retractall(command(0,({code},{color}), _))"))
                        
                    return True
                else:
                    self.prolog.assertz(f"walkable(({code},{color}), true)")
                    self.prolog.assertz(f"is_known(0,({code},{color}), true)")
                    list(self.prolog.query(f"retractall(command(0,({code},{color}), _))"))
            else:
                diff = self.hp[0]-blstats[10]
                results = list(self.prolog.query(f"risk_zone_symbol(({code},{color}), X)"))
                if len(results)>0:
                    if diff>0:
                        print("meno HP")
                        self.prolog.assertz(f"risk_zone_pos(({step[0]},{step[1]}),{diff})")
                        results = list(self.prolog.query(f"command(0,({code},{color}), X)"))
                        if len(results)==0:
                            self.prolog.assertz(f"command(0,({code},{color}), 1)")
                        else:
                            list(self.prolog.query(f"retractall(command(0,({code},{color}), _))"))
                            cont = (results[0]['X'])
                            self.prolog.assertz(f"command(0,({code},{color}), {cont+1})")
                        if cont >5:
                            self.prolog.assertz(f"risk_zone_symbol(({code},{color}), 10)")

                        return True
                else:
                    self.prolog.assertz(f"risk_zone_symbol(({code},{color}), 0)")
                    list(self.prolog.query(f"retractall(command(0,({code},{color}), _))"))
                    return False

        else:
            results = list(self.prolog.query(f"is_known({cmd},({code},{color}), X)"))
            if len(results)==0:

                results = list(self.prolog.query(f"command({cmd},({code},{color}), X)"))
                cont =0
                if len(results)==0:
                    self.prolog.assertz(f"command({cmd},({code},{color}), 1)")
                else:
                    list(self.prolog.query(f"retractall(command({cmd},({code},{color}), _))"))
                    cont = (results[0]['X'])
                    self.prolog.assertz(f"command({cmd},({code},{color}), {cont+1})")
                if cont >5:
                    self.prolog.assertz(f"is_known({cmd},({code},{color}), false)")

                tty_chars=self.obs['tty_chars']
                tty_colors=self.obs['tty_colors']

                obscode = tty_chars[step[1]][step[0]]
                obscolor = tty_colors[step[1]][step[0]]
                if ((color!=obscolor) or (code!=obscode)):
                    print(f"Debug: new")
                    self.prolog.assertz(f"is_known({cmd},({code},{color}), true)")
            
            return True


                    #self.prolog.assertz(f"is_useful({cmd},_({code},{color}), true)")

                ## Query: esiste command((0,0), X)?
                #results = list(prolog.query(f"command(0,({x},{y}), X)"))
                #print("action after assertz 2:", results)
#
                #self.prolog.assertz(f"risk_zone_pos(({x},{y}),100000)")
                #print(" QUI no ")
                #self.prolog.assertz(f"risk_zone_pos(({x},{y}),100000)")
                #self.pos = (blstats[0].item(), blstats[1].item())
                #return True

        self.pos = (blstats[0].item(), blstats[1].item())

        # Strength
        if self.strength > blstats[3]:
            print("meno forza")
        if self.strength < blstats[3]:
            print("più forza")
        self.strength = blstats[3]

        # Dexterity
        if self.dexterity > blstats[4]:
            print("meno destrezza")
        if self.dexterity < blstats[4]:
            print("più destrezza")
        self.dexterity = blstats[4]

        # Constitution
        if self.constitution > blstats[5]:
            print("meno costituzione")
        if self.constitution < blstats[5]:
            print("più costituzione")
        self.constitution = blstats[5]

        # Intelligence
        if self.intelligence > blstats[6]:
            print("meno intelligenza")
        if self.intelligence < blstats[6]:
            print("più intelligenza")
        self.intelligence = blstats[6]

        # Wisdom
        if self.wisdom > blstats[7]:
            print("meno saggezza")
        if self.wisdom < blstats[7]:
            print("più saggezza")
        self.wisdom = blstats[7]

        # Charisma
        if self.charisma > blstats[8]:
            print("meno carisma")
        if self.charisma < blstats[8]:
            print("più carisma")
        self.charisma = blstats[8]

        # Score
        if self.score > blstats[9]:
            print("meno punti")
        if self.score < blstats[9]:
            print("più punti")
        self.score = blstats[9]

        # HP (current only)
        if self.hp[0] > blstats[10]:
            print("meno HP")
        if self.hp[0] < blstats[10]:
            print("più HP")
        self.hp = (blstats[10], blstats[11])

        # Energy (current only)
        if self.energy[0] > blstats[14]:
            print("meno energia")
        if self.energy[0] < blstats[14]:
            print("più energia")
        self.energy = (blstats[14], blstats[15])

        # AC (armor class, qui più basso è meglio)
        if self.ac > blstats[16]:
            print("migliore classe armatura (più bassa)")
        if self.ac < blstats[16]:
            print("peggiore classe armatura (più alta)")
        self.ac = blstats[16]
        return False


