:- dynamic risk_zone_symbol/2.
:- dynamic risk_zone_pos/2.
:- dynamic walkable/2.
:- dynamic health/1.
:- dynamic health_max/1.
:- dynamic has/4.
:- dynamic goal/3.
:- dynamic agent_pos/2.
:- dynamic action/1.
:- dynamic near_goal/0.
:- dynamic openable/2.
:- dynamic hunger/1.
:- dynamic command/3.
:- dynamic is_known/3.
:- dynamic is_useful/4.
:- dynamic is_monster/2.
:- dynamic moveInvalid/3.
:- dynamic winner/2.


healthy :-
    health(H),
    health_max(M),
    Soglia is M * 0.75,
    H > Soglia.

hungry :- 
    hunger(X), 
    X > 1.


% Predicato: sei vicino al goal secondo la distanza di Chebyshev?
near_goal :-
    agent_pos(Xa, Ya),
    goal(pos(Xg, Yg), _, _),
    DX is abs(Xa - Xg),
    DY is abs(Ya - Yg),
    D is max(DX, DY),
    D =:= 1.


% Wrapper Python-style:

% Action drink
action(dict{cmd:64, slot:Key}) :-  % 64=quaff, slot=Key
    has(potion, healing, _, Key),
    \+ healthy.

% Azione: Open
action(dict{cmd:57}) :-
    near_goal,
    goal(_, X, Y),
    openable(X, Y).

% Action eat
action(dict{cmd:35, slot:Key}) :-  % 64=quaff, slot=Key
    has(food, _, _, Key),
    \+ hungry.

% Azione: Open
action(dict{cmd:57}) :-
    near_goal,
    goal(_, X, Y),
    openable(X, Y).




%walkable((46,7), true).
%walkable((43,3), false).
%walkable((45,7), false).
%walkable((124,7), false).

%openable(43, 3).
%openable(4, 3).

health(10).
health_max(10).
hunger(2).

has(potion, healing, _, b).
% Stato attuale del agente (posizione)
agent_pos(2,1).

% Goal (posizione da raggiungere)
goal(pos(1,1), 4, 3).