from pyswip import Prolog
from pathlib import Path
from utility import a_star, save, load, ascii_to_idx, inventoryToProlog, print_inventory, quote
import os
import time
import numpy as np
import sys

sys.path.append('/content/Progetto-AIF')
class AgentNetHack:
    def load(self):
        if Path("/content/Progetto-AIF/memory").exists():
            load("/content/Progetto-AIF/memory", self.prolog)
    def save(self):
        self.pointweapon()
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
        arr.append("weapon(X,Y)")
        with open("memory", "w") as f:
            for e in arr:
                str = e.split("(")
                f.write(save(e,str[0],self.prolog))
        list(self.prolog.query(f"retractall(risk_zone_symbol(_,_))"))
        list(self.prolog.query(f"retractall(walkable(_,_))"))
        list(self.prolog.query(f"retractall(openable(_,_))"))
        list(self.prolog.query(f"retractall(command(_,_,_))"))
        list(self.prolog.query(f"retractall(is_useful(_,_,_))"))
        list(self.prolog.query(f"retractall(is_monster(_,_))"))
        list(self.prolog.query(f"retractall(winner(_,_))"))
        list(self.prolog.query(f"retractall(weapon(_,_))"))
        list(self.prolog.query(f"retractall(is_known(_,_,_))"))
        list(self.prolog.query(f"retractall(risk_zone_pos(_,_))"))
        list(self.prolog.query(f"retractall(moveInvalid(_,_,_))"))
        list(self.prolog.query(f"retractall(has(_,_,_,_,_,_))"))
        
    
    def pointweapon(self):
        if self.weapon is not None:
            r = list(self.prolog.query(f"weapon({self.weapon},X)"))
            p = r[0]['X']
            if self.survivor:
                p = p+1 if p<10 else p
            else:
                p = p-1 if p>0 else p
            
            list(self.prolog.query(f"retractall(weapon({self.weapon},_))"))
            self.prolog.assertz(f"weapon({self.weapon}, {p})")

    def assertzIsKnown(self, cmd, code, color, flag):
        list(self.prolog.query(f"retractall(is_known({cmd},({code},{color}), {flag}))"))
        self.prolog.assertz(f"is_known({cmd},({code},{color}), {flag})")



    def __init__(self, env, pathProlog):
        self.prolog = Prolog()
        self.prolog.consult(pathProlog)
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
        self.hp = blstats[10]  # current, max
        self.hpold = self.hp
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

        self.unknow = set()
        self.explored = {}
        self.goals = []
        self.turni=0
        self.cachewalkable={}
        self.cardinal_directions =True
        self.cmd=[0,ascii_to_idx('o'),ascii_to_idx('c')] #, ascii_to_idx('F'),48]
        self.combatMode=False
        self.fight=True
        self.survivor =False
        self.monster= None
        self.healthy=True
        self.weapon = None
        self.terminal =False
        self.truncated =False

        height = len(self.obs['tty_chars']) - 3
        width = len(self.obs['tty_chars'][0])
        tuple_dtype = np.dtype([('x', np.int32), ('y', np.int32)])

        self.map = np.zeros((width, height), dtype=tuple_dtype)

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
                

        #print("Map")       
        #for y in range(height):
        #    for x in range(width):
        #        print(f"{chr(self.map[x][y]['x'])}", end="")
        #    print("")
        #print("end map")




    def goal(self, obj = None, reset=True):
        """
        Returns a list with symbol, position, and command.
        """
        self.combatMode=False
        self.goals=[]
        list(self.prolog.query(f"retractall(pos_monster(_,_))"))
            

        for y in range(self.height):
            for x in range(self.width):
                code = self.map[x][y]["x"].item()
                color = self.map[x][y]["y"].item()
               
                if self.explored.get((x,y),0)==1:
                    continue
                dist  = max(abs(x - self.pos[0]), abs(y - self.pos[1]))
                
                match (code, color):
                    case (0,0): #fog
                        self.goals.append(((code, color), (x, y), 1, dist, 0))
                    case (32,0): #wall
                        pass
                    case (64,15): #character
                        pass
                    case _:
                        results = list(self.prolog.query(f"walkable(({code},{color}), X)"))
                        if len(results) > 0:
                            continue

                        results = list(self.prolog.query(f"winner({code}, {color})"))
                        if len(results) > 0:
                            self.goals.append(((code, color), (x, y), 6, dist, 0))
                            continue
                        results = list(self.prolog.query(f"beliefSeeMonster(({code},{color}),{self.hp},{self.hpold},X)"))
                        if len(results) > 0:
                            danger = int(results[0]['X'])
                            self.goals.append(((code, color), (x, y), 6 + danger, dist, 1))
                            self.combatMode=True
                            list(self.prolog.query(f"retractall(is_known(_,({code},{color}),_))"))
                            list(self.prolog.query(f"retractall(command(_,({code},{color}),_))"))
                            list(self.prolog.query(f"retractall(is_useful(_,({code},{color}),_))"))
                            list(self.prolog.query(f"retractall(walkable(({code},{color}),_))"))
                            continue
                        
                        results = list(self.prolog.query(f"is_known(Y,({code},{color}), X)"))
                        if len(results) < len(self.cmd):
                            
                            self.goals.append(((code, color), (x, y), 3, dist, 2))
                            continue
                        results = list(self.prolog.query(f"is_useful(Y,({code},{color}), X)"))
                        if len(results) > 0:
                            priority = int(results[0]['X'])
                            self.goals.append(((code, color), (x, y), priority, dist, int(results[0]['Y'])))
                            
        self.goals = sorted(self.goals, key=lambda x: (-x[2], x[3]))
        if len(self.goals)==0:
            self.survivor =True
            return []
        print(f"goal {self.goals[0]}")


        if len(self.goals) ==0:
            print(f"You win")
            return []
        
        target = self.goals[0]

        list(self.prolog.query(f"retractall(goal(_,_,_))"))
        self.prolog.assertz(f"goal(({target[0][0]},{target[0][1]}),{target[1][0]}, {target[1][1]})")
        results1 = list(self.prolog.query(f"goal(Z, X,Y)"))

        if target[4]==2: # symbol unknown
            results = list(self.prolog.query(f"is_known(Y,({target[0][0]},{target[0][1]}), X)"))
            tmpcmd=self.cmd.copy()
            for i in results:
                if i.get('Y') in tmpcmd:
                    tmpcmd.remove(i.get('Y'))
            target = (target[0], target[1], target[2], target[3], tmpcmd[0])
        if target[4]==1: # there is a monster
            self.prolog.assertz(f"maybe_monster(({target[0][0]},{target[0][1]}),5)")
            self.cardinal_directions=False
            
            lista = self.neighbors(target[1])
            
            nodo = min(
                    lista,
                    key=lambda neighbor: max(abs(self.pos[0] - neighbor[0]), abs(self.pos[1] - neighbor[1]))
                )
            
            if nodo!=self.pos:
                path = a_star(self.pos, nodo, self)
            else:
                path = [self.pos]

        else:      
        
            if self.is_walkable(target[1], True):
                path = a_star(self.pos, target[1], self)

            else:
                lista = self.neighbors(target[1], cardinal_directions=True)
                nearestNeighbor = min(
                    lista,
                    key=lambda neighbor: max(abs(self.pos[0] - neighbor[0]), abs(self.pos[1] - neighbor[1]))
                )

                path = []
                if nearestNeighbor!=self.pos:
                    path = a_star(self.pos, nearestNeighbor, self)
                else:
                    path = [self.pos]
        if path is None:
            return []
        
        start = self.pos

        arr = []
        if len(path) ==1:
            cmd = []
            list(self.prolog.query(f"retractall(agent_pos(_,_))"))
            self.prolog.assertz(f"agent_pos({self.pos[0]},{self.pos[1]})")
            results1 = list(self.prolog.query(f"action(({target[0][0]},{target[0][1]}),{target[4]}, X)"))
            target = (target[0], target[1], target[2], target[3], results1[0]['X']['cmd'])
            n=1
            match target[4]:
                case 1: # fight
                    cmd.append(ascii_to_idx('F'))
                    cmd.append(self.move_to(start, target[1]))
                case 0:
                    self.fight = True
                    cmd.append(self.move_to(start, target[1]))
                case _:
                    cmd.append(target[4])
                    cmd.append(self.move_to(start, target[1]))
                    n=5             
            arr.append(((self.map[target[1][0]][target[1][1]]['x'].item(),self.map[target[1][0]][target[1][1]]['y'].item()), target[1], cmd, n))

        for step in path[1:]:
            cmd = []
            if step == target[1]:
                list(self.prolog.query(f"retractall(agent_pos(_,_))"))
                self.prolog.assertz(f"agent_pos({start[0]},{start[1]})")
                results1 = list(self.prolog.query(f"action(({target[0][0]},{target[0][1]}),{target[4]}, X)"))
                target = (target[0], target[1], target[2], target[3], results1[0]['X']['cmd'])
                match target[4]:
                    case 0:
                        cmd.append(self.move_to(start, step))
                    case _:
                        cmd.append(target[4])
                        cmd.append(self.move_to(start, step))
                n = 1 if target[0] ==(0,0) else 5                
                arr.append(((self.map[step[0]][step[1]]['x'].item(),self.map[step[0]][step[1]]['y'].item()), step, cmd, n))
            else:
                cmd.append(self.move_to(start, step))
                arr.append(((self.map[step[0]][step[1]]['x'].item(),self.map[step[0]][step[1]]['y'].item()), step, cmd, 1))
                start = step
        

        self.cardinal_directions=True

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

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.is_walkable((nx, ny), flag):
                        result.append((nx, ny))
            if len(result)>0:
                return result

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            # Check if the neighbor is not walkable
            
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.is_walkable((nx, ny), flag):
                    result.append((nx, ny))
        return result

    def move(self):

        inv = print_inventory(self.obs)
        path = self.goal()
        if self.survivor:
            return self.survivor
        if len(path) ==0:
            return self.survivor
        for symbol, pos, listcmd, n in path:
            symbol = (self.map[pos[0]][pos[1]]['x'].item(),self.map[pos[0]][pos[1]]['y'].item())
            for i in range(n):
                for cmd in listcmd:
                    self.turni += 1
                    if self.callStep(cmd, symbol):
                        return self.survivor
                flag = self.observe_and_update(symbol, pos, listcmd[0])
                if flag or listcmd[0]<8:
                    break
            if flag:
                break
        return self.move()
    
    def callStep(self,cmd, symbol):
        """Call the step method of MiniHack and check if it is terminal."""
        if self.terminal or self.truncated:
            return True
        self.obs, reward, self.terminal, self.truncated, info = self.env.step(cmd)
        self.env.render()
        if self.terminal or self.truncated:
            if reward > 0.5:
                self.survivor = True
                results = list(self.prolog.query(f"winner({symbol[0]}, {symbol[1]})"))
                if len(results) == 0:
                    self.prolog.assertz(f"winner({symbol[0]}, {symbol[1]})")
            return True
        return False
            

    def learnsymbol(self, symbol, pos, cmd):
        """Updates the knowledge base about the environment after the interaction."""
        flagUpdate  = False
        results = list(self.prolog.query(f"is_known({cmd},({symbol[0]},{symbol[1]}), X)"))
        if len(results)==0: # if agent don't know the symbol
            results = list(self.prolog.query(f"command({cmd},({symbol[0]},{symbol[1]}), X)"))
            cont=0
            if len(results)==0:
                self.prolog.assertz(f"command({cmd},({symbol[0]},{symbol[1]}), 1)")
            else:
                list(self.prolog.query(f"retractall(command({cmd},({symbol[0]},{symbol[1]}), _))"))
                cont = (results[0]['X'])
                self.prolog.assertz(f"command({cmd},({symbol[0]},{symbol[1]}), {cont+1})")
            if cont>5:
                if cmd ==0:
                    self.prolog.assertz(f"walkable(({symbol[0]},{symbol[1]}), false)")
                    flagUpdate = True
                    key = (symbol[0],symbol[1])
                    self.cachewalkable[key]=False
                self.assertzIsKnown(cmd, symbol[0], symbol[1], 'false')
                list(self.prolog.query(f"retractall(command({cmd},({symbol[0]},{symbol[1]}), _))"))
            
            results = list(self.prolog.query(f"is_known({cmd},({symbol[0]},{symbol[1]}), X)"))
            tty_chars=self.obs['tty_chars']
            tty_colors=self.obs['tty_colors']
            obscode = tty_chars[pos[1]+1][pos[0]]
            obscolor = tty_colors[pos[1]+1][pos[0]]
            if len(results)==0 and ((symbol[1]!=obscolor) or (symbol[0]!=obscode)) and (obscode,obscolor)!=(64,15):
                if self.combatMode:
                    if len(list(self.prolog.query(f"is_monster(({symbol[0]},{symbol[1]}), X)")))==0:
                        self.prolog.assertz(f"is_monster(({symbol[0]},{symbol[1]}), 1)")
                    self.combatMode=False
                else:
                    if cmd==0:
                        self.prolog.assertz(f"walkable(({symbol[0]},{symbol[1]}), false)")
                    self.assertzIsKnown(cmd, symbol[0], symbol[1], 'true')
                    self.prolog.assertz(f"is_useful({cmd},({symbol[0]},{symbol[1]}), 1)")

                flagUpdate = True
            elif len(results)>0 and ((symbol[1]!=obscolor) or (symbol[0]!=obscode)):
                flagUpdate = True


        results = list(self.prolog.query(f"is_known({cmd},({symbol[0]},{symbol[1]}), X)"))
        blstats = self.obs['blstats']
        if len(results) ==0 and  cmd==0 and (self.pos == (blstats[0].item(), blstats[1].item())):
            self.prolog.assertz(f"risk_zone_pos(({pos[0]},{pos[1]}),10)")
            flagUpdate = True
        elif len(results) ==0 and cmd==0:
            self.prolog.assertz(f"walkable(({symbol[0]},{symbol[1]}), true)")
            self.assertzIsKnown(cmd, symbol[0], symbol[1], 'true')
            list(self.prolog.query(f"retractall(command({cmd},({symbol[0]},{symbol[1]}), _))"))
            flagUpdate = False
        elif len(results) >0 and  cmd==0 and (self.pos == (blstats[0].item(), blstats[1].item())):
            self.updateMap()
            
            if (self.map[pos[0]][pos[1]]['x'].item(),self.map[pos[0]][pos[1]]['y'].item())==symbol:
                self.prolog.assertz(f"moveInvalid(({self.pos[0]},{self.pos[1]}), ({pos[0]},{pos[1]}), 1)")
            flagUpdate = True
        
        return flagUpdate

    def observe_and_update(self, symbol, step, cmd=None):
        """Checks the state of the environment."""
        blstats = self.obs['blstats']
        self.hpold = self.hp
        self.hp = blstats[10]
        list(self.prolog.query(f"retractall(health_max(_))"))
        list(self.prolog.query(f"retractall(health(_))"))
        self.prolog.assertz(f"health_max({blstats[11]})")
        self.prolog.assertz(f"health({blstats[10]})")
        msg = "".join([chr(c) for c in self.obs["message"]]).strip()
        if msg.find("miss")!=-1 or msg.find("hit")!=-1:
            self.explored={}
        if msg.find("[ynq]")!=-1:
            if self.callStep(ascii_to_idx('n'), symbol):
                return self.survivor
            self.explored[step]=1
        if msg[1:4]==" - ":
            descr = msg[6:]
            descr = descr[0:descr.find(".")]
            descr = quote(descr)

            if len(list(self.prolog.query(f"weapon({descr},X)")))==0:
                weapon = msg[0]
                if self.callStep(ascii_to_idx('w'), symbol):
                    return self.survivor
                msg = "".join([chr(c) for c in self.obs["message"]]).strip()
                msg =msg[msg.find('['):]
                msg =msg[msg.find(' or'):]
                if msg.find(weapon)!=-1:
                    self.prolog.assertz(f"weapon({descr}, 10)")
                    print("ARMAAAAAAAAAAA")
                    if self.callStep(ascii_to_idx(weapon), symbol):
                        return self.survivor
                    self.env.render()
                else:
                    if self.callStep(38, symbol):
                        return self.survivor
                    self.env.render()
            
            inv = print_inventory(self.obs)
            list(self.prolog.query(f"retractall(has(_,_,_,_,_,_))"))
            inventoryToProlog(inv, self.prolog)
            if  len(list(self.prolog.query(f"is_useful({0},({symbol[0]},{symbol[1]}), _)")))==0:
                self.prolog.assertz(f"is_useful({0},({symbol[0]},{symbol[1]}), 6)")
                for c in self.cmd:
                    self.assertzIsKnown(c, symbol[0], symbol[1], 'false')

            flagUpdate = True


        else:
            cmd = cmd if cmd >=8 else 0
            flagUpdate  = self.learnsymbol(symbol, step, cmd)
            flagUpdate = True if self.hp<self.hpold else False
        self.updateMap()
        
        list(self.prolog.query(f"retractall(has(_,_,_,_,_,_))"))
        inv = print_inventory(self.obs)
        inventoryToProlog(inv, self.prolog)
        ris = list(self.prolog.query(f"use(X,Y)"))
        print(f"ris {list(self.prolog.query('has(X,_,_,_,_,Y)'))}")
        for el in ris:
            if self.callStep(ascii_to_idx(el['Y']), symbol):
                        return self.survivor
            self.env.render()
            if self.callStep(ascii_to_idx(el['X']), symbol):
                        return self.survivor
            self.env.render()
            if el['Y']=='w':
                arma = list(self.prolog.query(f"has(Y,_,_,_,_,{el['X']})"))
                self.weapon = arma[0]['Y']
        list(self.prolog.query(f"retractall(has(_,_,_,_,_,_))"))

        self.pos = (blstats[0].item(), blstats[1].item())
        list(self.prolog.query(f"retractall(agent_pos(_,_))"))
        self.prolog.assertz(f"agent_pos({self.pos[0], self.pos[1]})")
        return flagUpdate
