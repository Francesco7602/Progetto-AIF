from pyswip import Prolog
from pathlib import Path
from utility import a_star, Simboli_unici, SymbolToPos, save, load, ascii_to_idx
import os
import time
import numpy as np

class AgentNetHack:
    def load(self):
        if Path("memory").exists():
            load("memory", self.prolog)
    def save(self):
        list(self.prolog.query(f"retractall(walkable((32,0),_))"))
        list(self.prolog.query(f"retractall(walkable((0,0),_))"))
        arr=[]
        arr.append("risk_zone_symbol(X,Y)")
        arr.append("walkable(X,Y)")
        arr.append("openable(X,Y)")
        arr.append("command(X,Y,Z)")
        arr.append("is_known(X,Y,Z)")
        arr.append("is_useful(X,Y,Z)")
        arr.append("is_monster(X,Y)")
        arr.append("winner(X,Y)")
        with open("memory", "w") as f:
            for e in arr:
                str = e.split("(")
                f.write(save(e,str[0],self.prolog))
        

    def __init__(self, env):
        self.prolog = Prolog()
        self.prolog.consult("kb.pl")
        self.load()
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

        self.height = len(self.obs["tty_chars"])-3
        self.width = len(self.obs["tty_chars"][0])
        #print(f"nel costruttore height: {self.height} e width: {self.width}")

        self.unknow = set()
        self.explored = {}
        self.goals = []
        self.turni=0
        self.cachewalkable={}
        self.cardinal_directions =True
        self.cmd=[0,ascii_to_idx('o')]

        height = len(self.obs['tty_chars']) - 3
        width = len(self.obs['tty_chars'][0])
        tuple_dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        #self.map = np.full((width, height), dtype=tuple_dtype)
        self.map = np.zeros((width, height), dtype=tuple_dtype)#dtype=np.int32 se da noia si rimette ma non intero
        #for x in range(0,width):
        #    for y in range(0,height):
        #        self.map[x][y]=(32,0)
        self.updateMap(True)

    
    def set_fog(self,x,y):
        """Set fog in the agent's internal map."""
        codeM = self.map[x][y]['x'].item()
        colorM = self.map[x][y]['y'].item()
        if codeM == 32 and colorM == 0:
            list = self.neighbors((x, y), flag=False)  # Find all neighbors that can be walked to
            if len(list) > 0:
                self.map[x][y] = (0, 0) # Set fog
            else:
                self.map[x][y] = (32, 0) # Set wall or object unknown

    
    def updateMap(self, init:bool=False):
        """
        updateMap builds and updates the agent's internal map.
        """

        tty_chars = self.obs['tty_chars']
        tty_colors = self.obs['tty_colors']

        # Remove the useless row
        height = len(tty_chars)
        tty_chars = tty_chars[1:height - 2]
        tty_colors = tty_colors[1:height - 2]

        height = len(tty_chars)
        width = len(tty_chars[0])

        for y in range(height):
            for x in range(width):
                code = tty_chars[y][x].item()
                color = tty_colors[y][x].item()
                if (code, color) == (46, 8):
                    color = 7
                codeM = self.map[x][y]['x'].item()
                colorM = self.map[x][y]['y'].item()
                if init or ((code, color) != (32, 0)) or ((code, color) != (46, 8)):
                    self.map[x][y] = (code, color)
                elif ((code, color) == (32, 0)) and (codeM, colorM) == (0, 0):
                    self.map[x][y] = (code, color)

        for y in range(height):
            for x in range(width):
                self.set_fog(x,y)
                

        print("Map")       
        for y in range(height):
            for x in range(width):
                print(f"{chr(self.map[x][y]['x'])}", end="")
            print("")
        print("end map")




    def goal(self, obj = None, reset=True):
        # returns list of (simbol, pos, list cmd)
        """
        Quando ritorni la lista di goal, torna pure le azioni consigliate su tutti o il primo
        cosi quando vedi una porta aperta gli dici di attraversarla
        quando fa un azione sul goal se è walkable dovre camminarci sopra e andare in unalta direzione che non sia quella da cui viene
        """

        self.goals=[]
        #target_pos = (self.pos[0], self.pos[1])
        #self.goals = [g for g in self.goals if g[1] != target_pos]
        #arr = []

        #if reset or obj:
        #    #print(f"DENTRO 1 {obj[0]}")
        #    if reset or (obj[0] ==(0,0)):
        #        print("DENTRO 2")
        #        self.goals=[]

        #height = len(self.map[0])
        #width = len(self.map)
        #print(f"ALTEZZA E LARGHEZZA {height} {width}")

        for y in range(self.height):
            for x in range(self.width):
                if self.explored.get((x,y),0)==1: #per evitare di andare nelle posizioni che ci danno errore (se non li conosciamo)
                    continue
                dist  = max(abs(x - self.pos[0]), abs(y - self.pos[1]))
                code = self.map[x][y]["x"].item()
                color = self.map[x][y]["y"].item()
                match (code, color):
                    case (0,0): #fog
                        self.goals.append(((code, color), (x, y), 4, dist, 0))
                    case (32,0): #wall
                        pass
                    case (64,15): #character
                        pass
                    case _:
                        #results = list(self.prolog.query(f"winner({code}, {color})"))
                        #if len(results) > 0:
                        #    self.goals.append(((code, color), (x, y), 100))
                        results = list(self.prolog.query(f"is_monster(({code},{color}), X)"))
                        if len(results) > 0:
                            danger = int(results[0]['X'])
                            self.goals.append(((code, color), (x, y), 6 + danger, dist, 1))
                            continue
                        results = list(self.prolog.query(f"is_known(Y,({code},{color}), X)"))
                        if len(results) != len(self.cmd):
                            self.goals.append(((code, color), (x, y), 3, dist, 2))
                            continue
                        results = list(self.prolog.query(f"is_useful(Y,({code},{color}), X)"))
                        if len(results) > 0:
                            priority = int(results[0]['X'])
                            self.goals.append(((code, color), (x, y), priority, dist, int(results[0]['Y'])))
        self.goals = sorted(self.goals, key=lambda x: (-x[2], x[3]))
        print(f"self.goals {self.goals}")

        print(f"goal {self.goals[0]}")

        #e = input("premi e per uscire .... ")
        #if e =='e':
        #    return []

        if len(self.goals) ==0:
            print(f"You win")
            return []
        
        target = self.goals[0]
        if target[4]==2:
            print (f"I need to learn ({target[0][0]},{target[0][1]})")
            results = list(self.prolog.query(f"is_known(Y,({target[0][0]},{target[0][1]}), X)"))
            #self.cardinal_directions =True
            target = (target[0], target[1], target[2], target[3], 0)
        if target[4]==1:
            print ("I need to fight")
        
        if target[4]==0 and self.cachewalkable.get(target[0],False):
            path = a_star(self.pos, target[1], self)
            #self.cardinal_directions =False
        else:
            lista=self.neighbors(target[1], cardinal_directions=True)
            print(f"{target[1]} non walkable: vado in {lista[0]}, {target[0]}")
            path = a_star(self.pos, lista[0], self)
            if len is None:
                path.append(target[1])
        if path is None:
            print("Not working")
            return []
        print(f"path: {path}")
        
        start = self.pos

        arr = []
        if len(path) ==1:
            print(f"start {start} path {path} target {target} ")
            cmd = []
            match target[4]:
                case 0:
                    cmd.append(self.move_to(start, target[1]))
                case _:
                    cmd.append(target[4])
                    cmd.append(self.move_to(start, target[1]))
            n = 1 if target[0] ==(0,0) else 5                
            arr.append(((self.map[target[1][0]][target[1][1]]['x'].item(),self.map[target[1][0]][target[1][1]]['y'].item()), target[1], cmd, n))
            
        for step in path[1:]:
            cmd = []
            if step == target[1]:
                match target[4]:
                    case 0:
                        cmd.append(self.move_to(start, step))
                    case _:
                        cmd.append(target[4])
                        cmd.append(self.move_to(start, step))
                n = 1 if target[0] ==(0,0) else 5                
                arr.append(((self.map[step[0]][step[1]]['x'].item(),self.map[step[0]][step[1]]['y'].item()), step, cmd, n))
            else:
                print(f"start: {start}, step: {step}, {self.move_to(start, step)}")
                cmd.append(self.move_to(start, step))
                arr.append(((self.map[step[0]][step[1]]['x'].item(),self.map[step[0]][step[1]]['y'].item()), step, cmd, 1))
                start = step
        

        print(f"arr: {arr}")

        #e = input("premi e per uscire .... ")
        #if e =='e':
        #    return []

        return arr
                
        



    def move_to(self, pos_cur, pos_prox):
        """
        move_to returns a command value that represents the direction from pos_cur to pos_prox.
        """

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
    
    def heuristic(self, current, pos, goal):
        """
        Compute the heuristic using the Chebyshev metric and apply logical_penalty to penalize passing through a dangerous area.
        """

        x, y = pos
        x_goal, y_goal = goal

        h = max(abs(x - x_goal), abs(y - y_goal)) + self.logical_penalty(current, pos)
        return h
    
    def logical_penalty(self, current, pos):
        """
        logical_penalty returns a penalty value for the given position.
        """

        term = 0
        x, y = pos

        code = self.map[x][y]['x']
        color = self.map[x][y]['y']

        results=list(self.prolog.query(f"risk_zone_symbol(({code},{color}), X)"))
        
        if len(results)>0:
            term = int(results[0]['X'])
        results=list(self.prolog.query(f"risk_zone_pos(({x},{y}), X)"))
        
        if len(results)>0:
            term += int(results[0]['X'])

        results = list(self.prolog.query(f"moveInvalid(({current[0]},{current[1]}), ({x},{y}), X)"))
        if len(results)>0:
            term += int(results[0]['X'])

        
        return term


    def is_walkable(self, pos, flag):
        """
        is_walkable returns True if the position is walkable.
        It's necessary to manage self.cachewalkable.
        """
        x, y = pos
        code = self.map[x][y]['x'].item()
        color = self.map[x][y]['y'].item()

        if flag and (code, color) == (0 ,0):# or (code, color) == (43, 3)):
           return True

        # Use cachewalkable to improve performance
        key = (code,color)
        if key not in self.cachewalkable:
            results=list(self.prolog.query(f"walkable(({code},{color}), X)"))
            if len(results)==0:
                self.cachewalkable[key]=True
            else:
                res = results[0]["X"] == 'true'
                self.cachewalkable[key]=res
                return res
        return self.cachewalkable[key]
        

    def neighbors(self, pos, flag = True, cardinal_directions=False):
        """
        Find all neighbors that can be walked to.
        If the flag is True, tiles covered by fog are also considered walkable.
        """
        cardinal_directions = True if self.cardinal_directions else cardinal_directions
        x , y = pos
        result = []
        if cardinal_directions:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                # Check if the neighbor is not walkable
                #results = list(self.prolog.query(f"moveInvalid(({x},{y}), ({nx},{ny}), X)"))
                #if len(results)>0:
                #    continue
            
                if 0 <= nx < self.width and 0 <= ny < self.height:

                    if self.is_walkable((nx, ny), flag):
                        result.append((nx, ny))
            if len(result)>0:
                return result

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            # Check if the neighbor is not walkable
            #results = list(self.prolog.query(f"moveInvalid(({x},{y}), ({nx},{ny}), X)"))
            #if len(results)>0:
            #    continue
            
            if 0 <= nx < self.width and 0 <= ny < self.height:
                
                if self.is_walkable((nx, ny), flag):
                    result.append((nx, ny))
        return result

    def move(self):
        path = self.goal()
        if len(path) ==0:
            return
        for simbol, pos, listcmd, n in path:
            simbol = (self.map[pos[0]][pos[1]]['x'].item(),self.map[pos[0]][pos[1]]['y'].item())
            print(f" Simbol: {simbol}, position: {pos} ")
            for i in range(n):
                for cmd in listcmd:
                    self.turni += 1
                    print(f" cmd {cmd} ")
                    self.obs, reward, terminal, truncated, info = self.env.step(cmd)
                    self.env.render()
                    #time.sleep(0.5)
                    if terminal or truncated:
                        if reward > 0.5:

                            results = list(self.prolog.query(f"winner({simbol[0]}, {simbol[1]})"))
                            if len(results) == 0:
                                self.prolog.assertz(f"winner({simbol[0]}, {simbol[1]})")
                        print(f"winner {simbol} bravo {reward} {type(reward)}")
                        return
                flag = self.observe_and_update(simbol, pos, listcmd[0])
                if flag or listcmd[0]<8:
                    break
            if flag:
                break
        return self.move()
            


    def move2(self, objgoal=None):
        self.pos = (self.obs["blstats"][0].item(), self.obs["blstats"][1].item())
        #self.updateMap()
        
        self.turni+=1
        start = (self.pos[0], self.pos[1])
        #os.system('clear')  # Pulisce il terminale
        self.env.render()
        #time.sleep(0.5)  # Aspetta un po' per vedere il frame
        #self.goals = SymbolToPos(self.obs, self.prolog, self.explored, self.goals)
        self.goal(objgoal)#self.obs, self.prolog, self.explored, self.goals, self.turni
        print(f"goals: {self.goals}")
        #if self.turni>1:
        #    for y in range(self.height):
        #        for x in range(self.width):
        #            print(f"{chr(self.map[x][y]['x'])}", end="")
        #        print("")
        #    return
        #return
        
        #print(self.pos)
        #print("GOALLIST")
        #print(self.goals)
        #print(goals)
        path = None
        if self.goals[0][2] >= 6 and self.goals[0][2] < 100:
            print("Hahaha mostro")

        else:
            objgoal = self.goals[0]
            goal = self.goals[0][1]
            print(f"debug ... ({start} to {goal} {self.map[start[0]][start[1]]['x']} goal pos: {self.map[goal[0]][goal[1]]['x']})")
            #print(f"debug ... ({start} to {goal} {self.map[start[0]][start[1]-1]['x']} goal pos: {self.map[goal[0]][goal[1]]['x']})")
            #print(f"debug ... ({start} to {goal} {self.map[start[0]][start[1]+1]['x']} goal pos: {self.map[goal[0]][goal[1]]['x']})")

            print(f"debug @... ({start} to {goal} {self.obs['tty_chars'][start[1]+1][start[0]]})")
            
            #print(f"debug ... ({start} to {goal} {self.map[start[0]][start[1]]['x']} goal pos: {self.map[goal[0]][goal[1]]['x']})")
            #print(f"debug ... ({start} to {goal} {self.map[start[0]][start[1]]['x']} goal pos: {self.map[goal[0]][goal[1]]['x']})")
            #print(f"debug ... ({start} to {goal} {self.map[start[0]][start[1]]['x']} goal pos: {self.map[goal[0]][goal[1]]['x']})")
            
            results = list(self.prolog.query(f"walkable(({objgoal[0][0]},{objgoal[0][1]}), X)"))
            if len(results)>0 and results[0]['X']=='false':
                r=self.neighbors(goal[0],goal[1])
                print(f"non walkable: vado in {r[0]}, {objgoal[0]}")
                path = a_star(start, r[0], self, self.obs)
                path.append(goal)
                print(f"path del vic {path}")
            else:
                path = a_star(start, goal, self, self.obs)
            print (f"fun move: goal path: {path}")
        obj={}
        if path is None:
            return
            goal = self.goals[1][1]
            path = a_star(start, goal, self, self.obs)
        for step in path[1:]:
            
            self.pos = (self.obs["blstats"][0].item(), self.obs["blstats"][1].item())
            start = (self.pos[0], self.pos[1])

            #print("UpdateMap")
            #self.updateMap()

            print(f"debug character ... {chr(self.map[start[0]][start[1]]['x'])} ({start} to {goal} goal pos: {self.map[goal[0]][goal[1]]['x']})")
            print(f"debug characterOBS ... {chr(self.obs['tty_chars'][start[1]+1][start[0]])} ({start} to {goal} goal pos: {self.map[goal[0]][goal[1]]['x']})")
            self.explored[step]=1
            
            obj['code']=self.map[step[0]][step[1]]['x']
            obj['color']=self.map[step[0]][step[1]]['y']

            #obj['code']=self.obs['tty_chars'][step[1]][step[0]]
            #obj['color']=self.obs['tty_colors'][step[1]][step[0]]

            results = list(self.prolog.query(f"is_known(Y,({obj['code']},{obj['color']}), X)"))
            cmdlist = [ascii_to_idx('o'), ascii_to_idx('c')]

            print(f"Debug sono (start {start}) ({self.pos[0]},{self.pos[1]+1}) vado in ({step}) symbol: {chr(obj['code'])} ({obj['code']}, {obj['color']}), results {results}")
            if type(cmdlist) != list:
                cmdlist = [cmdlist]

            if len(results) == len(cmdlist)+1:
                tmpcmd = cmdlist
                for i in results:
                    if i.get('X')== 'false' and i.get('Y') in tmpcmd:
                        tmpcmd.remove(i.get('Y'))
            else:
                tmpcmd=cmdlist
                for i in results:
                    if i.get('Y') in tmpcmd:
                        tmpcmd.remove(i.get('Y'))

            #print("AZIONE APPLICATA A :")
            #print(f"debug {tmpcmd, obj['code'], obj['color']}")
            #if len(results)==0 or (len(results)>0 and results[0].get('X', 'true') == 'true'):
            if step == goal:
                
                if obj['code'] ==0:
                    self.updateMap()
                    self.move(objgoal)
                    return

                print(f"sono accanto al goal {obj['code']}")
                results = list(self.prolog.query(f"walkable(({obj['code']}, {obj['color']}), X)"))
                walkable = False if len(results)>0 and results[0]['X']=="false" else True
                if len(tmpcmd)==0:# and not walkable:#se non sai che fare, cerca un altro goal
                    if walkable:
                        print(f"Attualmente mi trovo in {self.pos}")
                        cmd = self.move_to(start, step)
                        self.obs, reward, terminal, truncated, info = self.env.step(cmd)
                        self.env.render()
                        #time.sleep(0.5)
                        if terminal or truncated:
                            if reward > 0.5:
                                results = list(self.prolog.query(f"winner({obj['code']}, {obj['color']})"))
                                if len(results) == 0:
                                    self.prolog.assertz(f"winner({obj['code']}, {obj['color']})")
                            print(f"bravo {reward} {type(reward)}")

                            return
                        print("Mi sono spostato e calcolo il nuovo goal")

                        self.observe_and_update(step, cmd=cmd, obj=obj)  # aggiorna KB con nuove info
                        print(f"Attualmente mi trovo in {self.pos}")
                        self.move()
                        return
                    print("nuovo goal 2")
                    self.move()
                    return
                #elif len(tmpcmd)==0:
                #    self.move()
                #    return
                for cmd in tmpcmd:
                    self.obs, reward, terminal, truncated, info = self.env.step(cmd)
                    #os.system('clear')  # Pulisce il terminale
                    #self.env.render()
                    #time.sleep(0.5)  # Aspetta un po' per vedere il frame
                    cmd1 = self.move_to(start, step)  # direction
                    self.obs, reward, terminal, truncated, info = self.env.step(cmd1)
                    self.env.render()
                    #time.sleep(0.5)
                    #os.system('clear')  # Pulisce il terminale
                    #self.env.render()
                    #time.sleep(0.5)  # Aspetta un po' per vedere il frame
                    self.observe_and_update(step, cmd=cmd, obj=obj)
                    

            else:
                cmd = self.move_to(start, step)
                self.obs, reward, terminal, truncated, info = self.env.step(cmd)
                self.env.render()
                #time.sleep(0.5)
                #self.env.render()



            #    cmd = self.move_to(start, step)  # esegui lo spostamento logico/fisico
            #self.obs, reward, terminal, truncated, info = self.env.step(cmd)
            #else:
            if terminal or truncated:
                if reward >0.5:
                    results = list(self.prolog.query(f"winner({obj['code']}, {obj['color']})"))
                    if len(results)==0:
                        self.prolog.assertz(f"winner({obj['code']}, {obj['color']})")
                print(f"bravo {reward} {type(reward)}")

                return
            #os.system('clear')  # Pulisce il terminale

            #time.sleep(0.5)  # Aspetta un po' per vedere il frame
            if self.observe_and_update(step, cmd=cmd, obj=obj):  # aggiorna KB con nuove info
                print("nuovo goal 1")
                self.move()
                return
            elif step == goal:
                self.move()
                return

    def learnSimbol(self, simbol, pos, cmd):
        flagUpdate  = False
        results = list(self.prolog.query(f"is_known({cmd},({simbol[0]},{simbol[1]}), X)"))
        if len(results)==0: # if agent don't know the simbol
            results = list(self.prolog.query(f"command({cmd},({simbol[0]},{simbol[1]}), X)"))
            cont=0
            if len(results)==0:
                self.prolog.assertz(f"command({cmd},({simbol[0]},{simbol[1]}), 1)")
            else:
                list(self.prolog.query(f"retractall(command({cmd},({simbol[0]},{simbol[1]}), _))"))
                cont = (results[0]['X'])
                self.prolog.assertz(f"command({cmd},({simbol[0]},{simbol[1]}), {cont+1})")
            if cont>5:
                if cmd ==0:
                    self.prolog.assertz(f"walkable(({simbol[0]},{simbol[1]}), false)")
                    flagUpdate = True
                    key = (simbol[0],simbol[1])
                    self.cachewalkable[key]=False
                self.prolog.assertz(f"is_known({cmd},({simbol[0]},{simbol[1]}), false)")
                list(self.prolog.query(f"retractall(command({cmd},({simbol[0]},{simbol[1]}), _))"))
            
        #if cmd !=0:
            results = list(self.prolog.query(f"is_known({cmd},({simbol[0]},{simbol[1]}), X)"))
            tty_chars=self.obs['tty_chars']
            tty_colors=self.obs['tty_colors']
            obscode = tty_chars[pos[1]+1][pos[0]]
            obscolor = tty_colors[pos[1]+1][pos[0]]
            if len(results)==0 and ((simbol[1]!=obscolor) or (simbol[0]!=obscode)) and (obscode,obscolor)!=(64,15):
                self.prolog.assertz(f"is_known({cmd},({simbol[0]},{simbol[1]}), true)")
                self.prolog.assertz(f"is_useful({cmd},({simbol[0]},{simbol[1]}), 1)")
                flagUpdate = True
                print(f"INFO: is_known({cmd},({simbol[0]},{simbol[1]}), true) new symbol {obscode} {obscolor}")
            elif len(results)>0 and ((simbol[1]!=obscolor) or (simbol[0]!=obscode)):
                flagUpdate = True


        results = list(self.prolog.query(f"is_known({cmd},({simbol[0]},{simbol[1]}), X)"))
        blstats = self.obs['blstats']
        if len(results) ==0 and  cmd==0 and (self.pos == (blstats[0].item(), blstats[1].item())):
            print(f"probabile muro {pos} element {simbol[0]} {simbol[1]}")
            self.prolog.assertz(f"risk_zone_pos(({pos[0]},{pos[1]}),10)")
            self.explored[pos]=1
            flagUpdate = True
        elif len(results) ==0 and cmd==0:
            print("Non lo conosco, ma ora ho capito che è walkable")
            self.prolog.assertz(f"walkable(({simbol[0]},{simbol[1]}), true)")
            self.prolog.assertz(f"is_known({cmd},({simbol[0]},{simbol[1]}), true)")
            list(self.prolog.query(f"retractall(command({cmd},({simbol[0]},{simbol[1]}), _))"))
            flagUpdate = False
        elif len(results) >0 and  cmd==0 and (self.pos == (blstats[0].item(), blstats[1].item())):
            self.updateMap()
            print(f"moveInvalid: {simbol}, {self.map[pos[0]][pos[1]]['x'].item(),self.map[pos[0]][pos[1]]['y'].item()}")

            if (self.map[pos[0]][pos[1]]['x'].item(),self.map[pos[0]][pos[1]]['y'].item())==simbol:
                self.prolog.assertz(f"moveInvalid(({self.pos[0]},{self.pos[1]}), ({pos[0]},{pos[1]}), 1)")
                print(f"insert moveInvalid {self.pos[0]}, {self.pos[1]} to ({pos[0]},{pos[1]})")
            flagUpdate = True
        
        return flagUpdate

    def observe_and_update(self, simbol, step, cmd=None, obj=None):
        blstats = self.obs['blstats']
        cmd = cmd if cmd >=8 else 0
        flagUpdate  = self.learnSimbol(simbol, step, cmd)
        self.updateMap()
        
        self.pos = (blstats[0].item(), blstats[1].item())
        return flagUpdate

        
    def observe_and_update2(self, step, cmd=None, obj=None):

        code = obj['code']
        color = obj['color']
        #if code ==0:
        #    self.updateMap()
        #    return True
        blstats = self.obs["blstats"]
        print(f"cmd {cmd}, mappa pos: {self.pos}")
        tty_chars =self.obs["tty_chars"]
        tty_colors =self.obs["tty_colors"]

        obscode = tty_chars[self.pos[1]+1][self.pos[0]]
        obscolor = tty_colors[self.pos[1]+1][self.pos[0]]
        self.map[self.pos[0]][self.pos[1]]=(obscode,obscolor)

        obscode = tty_chars[step[1]+1][step[0]]
        obscolor = tty_colors[step[1]+1][step[0]]
        self.map[step[0]][step[1]]=(obscode,obscolor)

        
        flagUpdate  = False


        if cmd<8:
            cont = 0
            if (self.pos == (blstats[0].item(), blstats[1].item())):
                self.prolog.assertz(f"moveInvalid(({self.pos[0]},{self.pos[1]}), ({step[0]},{step[1]}), 10)")
                print(f"insert moveInvalid {self.pos[0]}, {self.pos[1]} to ({step[0]},{step[1]})")
                self.explored[step]=0
                flagUpdate = True
                #return True
            results = list(self.prolog.query(f"is_known(0,({code},{color}), X)")) #e possibile camminarci sopra
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
                        self.prolog.assertz(f"is_known(0,({code},{color}), false)")
                        list(self.prolog.query(f"retractall(command(0,({code},{color}), _))"))
                        
                    return True
                else:
                    print("Non lo conosco, ma ora ho capito che è walkable")
                    self.prolog.assertz(f"walkable(({code},{color}), true)")
                    self.prolog.assertz(f"is_known(0,({code},{color}), true)")
                    list(self.prolog.query(f"retractall(command(0,({code},{color}), _))"))
                    return False
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
                    print("Dovrebbe essere una zona sicura")
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
                    print(f"INFO: is_known({cmd},({code},{color}), false)")

                tty_chars=self.obs['tty_chars']
                tty_colors=self.obs['tty_colors']

                obscode = tty_chars[step[1]+1][step[0]]
                obscolor = tty_colors[step[1]+1][step[0]]
                if ((color!=obscolor) or (code!=obscode)):
                    print(f"Debug: new")
                    self.map[step[0]][step[1]]=(obscode,obscolor)
                    self.prolog.assertz(f"is_known({cmd},({code},{color}), true)")
                    print(f"INFO: is_known({cmd},({code},{color}), true) new symbol {obscode} {obscolor}")

            
            self.updateMap()
            #self.goals=[]
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
        return flagUpdate
