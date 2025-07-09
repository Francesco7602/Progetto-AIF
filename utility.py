from pyswip import Prolog
from nle.nethack import actions
from colorama import Fore, Style, init
import numpy as np
import re

init(autoreset=True)  # Così i colori si resettano ad ogni stampa

# Mappa tra codice colore NetHack e colori Colorama (modifica se vuoi)
nethack_to_colorama = {
    0: Style.RESET_ALL,      # Black/background/default
    1: Fore.RED,
    2: Fore.GREEN,
    3: Fore.YELLOW,
    4: Fore.BLUE,
    5: Fore.MAGENTA,
    6: Fore.CYAN,
    7: Fore.WHITE,
    8: Style.BRIGHT + Fore.BLACK,   # Bold/gray
    9: Style.BRIGHT + Fore.RED,
    10: Style.BRIGHT + Fore.GREEN,
    11: Style.BRIGHT + Fore.YELLOW,
    12: Style.BRIGHT + Fore.BLUE,
    13: Style.BRIGHT + Fore.MAGENTA,
    14: Style.BRIGHT + Fore.CYAN,
    15: Style.BRIGHT + Fore.WHITE,
}


def save(pred, name, prolog):
    stringa=""
    results = list(prolog.query(pred))
    #print(f"{name} {pred} {results}")
    for item in results:
        stringa+=name+"("
        for key in item:
            tmp = item[key]
            if type(tmp)==int:
                tmp = str(tmp)
            pos = (tmp).find(",")
            if pos ==0:
                stringa+=tmp.replace(",","",1)+","
            else:
                stringa+=tmp+","
        stringa=stringa[:len(stringa)-1]
        stringa+=")\n"
    return stringa

def load(file, prolog):
    with open(file, "r") as f:
        str = f.read()
    list = str.split("\n")
    for s in list:
        if s.strip():  # Solo se la stringa non è vuota
            prolog.assertz(s)

def a_star(start, goal, agent):
    from heapq import heappush, heappop

    open_set = []  # priority queue: (f(n), position)
    heappush(open_set, (0, start))

    came_from = {}  # maps each node to its predecessor
    cost_so_far = {start: 0}  # g(n): cost from start to node

    print(f"[A*] Start: {start}, Goal: {goal}")

    while open_set:
        _, current = heappop(open_set)  # node with lowest f(n)

        if current == goal:
            # reconstruct path from goal to start
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nx, ny in agent.neighbors(current):  # get walkable neighbors
            neighbor = (nx, ny)
            new_cost = cost_so_far[current] + 1  # cost to reach this neighbor

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + agent.heuristic(current, neighbor, goal)  # f(n) = g(n) + h(n)
                heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    return None

def ascii_to_idx(char):
    """
    Dato un carattere (es 'e'), restituisce l'indice della corrispondente azione in actions.ACTIONS.
    """
    val = ord(char)
    for idx, action in enumerate(actions.ACTIONS):
        if hasattr(action, 'value') and action.value == val:
            return idx
    return None  # Non trovato

def print_stats(observation):
    blstats = observation['blstats']

    # Campi di blstats secondo la documentazione NLE:
    (y, x) = blstats[0], blstats[1]
    strength = blstats[3]
    dexterity = blstats[4]
    constitution = blstats[5]
    intelligence = blstats[6]
    wisdom = blstats[7]
    charisma = blstats[8]
    level = blstats[9]
    hp, max_hp = blstats[10], blstats[11]
    mana, max_mana = blstats[12], blstats[13]
    ac = blstats[14]
    gold = blstats[15]
    # blstats[16] = hunger
    hunger = blstats[16] if len(blstats) > 16 else None
    # blstats[17] = time
    turn = blstats[17] if len(blstats) > 17 else None
    # blstats[18] = score
    score = blstats[18] if len(blstats) > 18 else None
    # blstats[19] = XP
    exp = blstats[19] if len(blstats) > 19 else None
    # blstats[20] = alignment
    alignment = blstats[20] if len(blstats) > 20 else None
    # blstats[21] = status
    status = blstats[21] if len(blstats) > 21 else None
    # blstats[22] = monnum
    monnum = blstats[22] if len(blstats) > 22 else None
    # blstats[23] = gender
    gender = blstats[23] if len(blstats) > 23 else None
    # blstats[24] = armor
    armor = blstats[24] if len(blstats) > 24 else None

    print("---- CARATTERISTICHE PRINCIPALI ----")
    print(f"Posizione: ({y}, {x})")
    print(f"Livello dungeon: {level} | Turno: {turn}")
    print(f"HP: {hp}/{max_hp} | Mana: {mana}/{max_mana}")
    print(f"Oro: {gold} | Classe Armatura (AC): {ac} | Fame: {hunger}")
    print(f"Forza: {strength} | Des: {dexterity} | Cos: {constitution} | Int: {intelligence} | Sag: {wisdom} | Car: {charisma}")
    print(f"Esperienza: {exp} | Allineamento: {alignment} | Status: {status} | Mostro: {monnum} | Genere: {gender} | Armatura: {armor}")
    print(f"Punteggio: {score}")
    print("------------------------------------\n")

    ## Messaggio corrente e inventario (se esistono)
    #if len(obs) > 5 and isinstance(obs[5], (list, tuple)) or hasattr(obs[5], "__iter__"):
    #    msg = "".join(chr(c) for c in obs[5] if c != 0)
    #    print("Ultimo messaggio:", msg)
    #if len(obs) > 7 and isinstance(obs[7], (list, tuple)) or hasattr(obs[7], "__iter__"):
    #    inv_letters = [chr(c) for c in obs[7] if c != 0]
    #    print("Inventario (lettere):", inv_letters)
    print("------------------------------------\n")

def print_inventory(obs):
    inv_letters = obs["inv_letters"]
    inv_strs = obs["inv_strs"]
    array = []
    #print("Inventario:")
    for idx, c in enumerate(inv_letters):
        if c != 0:
            letter = chr(c)
            # Ogni inv_str è una lista di codici ASCII: convertiamola in stringa
            obj_name = "".join([chr(x) for x in inv_strs[idx] if x != 0]).strip()
            array.append(f"  {letter}: {obj_name}")
    #        print(f"  {letter}: {obj_name}")
    #print()
    return array




def Simboli_unici(observation):
    
    tty_chars=observation['tty_chars']
    tty_colors=observation['tty_colors']
    # Remove the useless row
    height = len(tty_chars)
    tty_chars = tty_chars[1:height-2]
    tty_colors = tty_colors[1:height-2]
    symbols = set()


    for row_chars, row_colors in zip(tty_chars, tty_colors):
        for char, color in zip(row_chars, row_colors):
            #if color!=0:
                symbols.add((char, color))

            

    print("\n------ Simboli unici ------")
    print(f"{'Simbolo':^8} | {'Codice':^6} | {'Colore':^6}")
    print("-" * 30)

    #debug
    for code, color in sorted(symbols):
        char = chr(code)
        color_code = nethack_to_colorama.get(color, Style.RESET_ALL)
        print(f"{color_code}{char:^8}{Style.RESET_ALL} | {code:^6} | {color:^6}")
    return symbols

def SymbolToPos(Map, prolog, dict, oldGoal= [], turni= 1):#todo levare questa funzione
    arr=[]

    tty_chars=Map['tty_chars']
    tty_colors=Map['tty_colors']

    # Remove the useless row
    height = len(tty_chars)
    tty_chars = tty_chars[1:height-2]
    tty_colors = tty_colors[1:height-2]

    height = len(tty_chars)
    width = len(tty_chars[0])

    for y in range(height):
        for x in range(width):
            if dict.get((x, y+1), 0) == 1:
                continue
            code =tty_chars[y][x].item()
            color = tty_colors[y][x].item()

            if oldGoal is not None:
                esiste = any(elem[1:2] == ((x,y)) for elem in oldGoal)
                if esiste == True:
                    #print(f"Esistono duplicati di {code} {color}, {x} {y}")
                    continue
            
            if (color==0 or (chr(code)=='@' and color==15)):
                continue
            results = list(prolog.query(f"winner({code}, {color})"))
            if len(results)>0:
                arr.append(((code,color),(x,y+1),100))
            results = list(prolog.query(f"is_monster(({code},{color}), X)"))
            if len(results)>0:
                danger = int(results[0]['X'])
                arr.append(((code,color),(x,y+1),6+danger+turni/200))
                continue
            results = list(prolog.query(f"walkable(({code},{color}), X)"))
            if len(results)>0:
                if results[0]['X']=='true':
                    arr.append(((code,color),(x,y+1),1+turni/200))
                    continue
            results = list(prolog.query(f"is_known(Y,({code},{color}), X)"))
            if len(results)==0:
                arr.append(((code,color),(x,y+1),5+turni/200))
            elif int(results[0]['Y'])==57: #is open
                arr.append(((code,color),(x,y+1),1+turni/200))


    Simboli_unici(Map)
    combined = arr + oldGoal
    """print("-----------SymbolToPos-----------")
    print(sorted(arr, key=lambda x: x[2], reverse= True))"""
    return sorted(combined, key=lambda x: x[2], reverse= True)

def inventoryToProlog2(list, prolog):
    for line in list:
        match = re.search(r"([a-z]): (an?|[\d]+)? ?(blessed|uncursed|cursed)? ?(\+\d+)? ?(.*?)( \((.*?)\))?$", line)
        if match:
            id, article, state, bonus, obj, _, note = match.groups()
            try:
                quantity = int(article)
            except:
                quantity = 1
            """print({
                "id": id,
                "object": obj,
                "state": state,
                "bonus": bonus,
                "quantity": quantity,
                "note": note,
            })"""
            flag = "false" if note is None else "true"
            prolog.assertz(f"has({obj}, {state}, {quantity}, {bonus}, {flag}, {id})")
def inventoryToProlog(lst, prolog):
    for line in lst:
        match = re.search(r"([a-z]): (an?|[\d]+)? ?(blessed|uncursed|cursed)? ?(\+\d+)? ?(.*?)( \((.*?)\))?$", line)
        if match:
            id, article, state, bonus, obj, _, note = match.groups()
            try:
                quantity = int(article)
            except:
                quantity = 1

            # Sanitize for Prolog
            def quote(s):
                if s is None:
                    return 'none'
                s = s.strip()
                # Metti apici solo se contiene spazi o simboli
                if re.search(r"[^\w]", s):
                    return f"'{s}'"
                return s

            # Se state o bonus sono None, metti 'none'
            state = quote(state)
            bonus = quote(bonus)
            obj = quote(obj)
            #print(f"inventoryToProlog: {obj}")
            flag = "false" if note is None else "true"
            prolog.assertz(f"has({obj}, {state}, {quantity}, {bonus}, {flag}, {id})")
