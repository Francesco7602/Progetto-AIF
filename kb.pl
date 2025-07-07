:- dynamic risk_zone_symbol/2.
:- dynamic risk_zone_pos/2.
:- dynamic walkable/2.
:- dynamic health/1.
:- dynamic health_max/1.
:- dynamic has/4.
% goal(simbol,pos) simbol=(code,color) and x,y
:- dynamic goal/3.
% agent_pos is position of agent x,y
:- dynamic agent_pos/2.
%action(simbol, cmd, D) verify command cmd of the simbol return dict or []
:- dynamic action/3.
% near_goal(S) return true if the agent is near the simbol
:- dynamic near_goal/1.
:- dynamic openable/2.
:- dynamic hunger/1.
:- dynamic command/3.
:- dynamic is_known/3.
:- dynamic is_useful/3.
:- dynamic is_monster/2.
:- dynamic moveInvalid/3.
:- dynamic winner/2.

% escape(X,Y) X simbol moster and Y true/false
:- dynamic escape/2.
%escape(_,false).

walkable((0, 0),false).
walkable((32, 0),false).

healthy :-
    health(H),
    health_max(M),
    Soglia is M * 0.75,
    H > Soglia.

hungry :- 
    hunger(X), 
    X > 1.

% Predicato: sei vicino al goal secondo la distanza di Chebyshev?
near_goal(S) :-
    agent_pos(Xa, Ya),
    goal(S, Xg, Yg),
    DX is abs(Xa - Xg),
    DY is abs(Ya - Yg),
    D is max(DX, DY),
    D =:= 1.

%% Action close
%action(S, C dict{cmd:C}) :-
%    escape(_,true),
%    is_useful(30,S, _),
%    near_goal(S).

action(S, 30, dict{cmd:30}) :-
    escape(_, true),
    is_useful(30, S, _),
    near_goal(S).

action(S, 30, dict{cmd:0}) :-
    escape(_, false),
    is_useful(30, S, _),
    near_goal(S).

action(S, C, dict{cmd:C}).



%% Wrapper Python-style:
%
%% Action drink
%action(dict{cmd:64, slot:Key}) :-  % 64=quaff, slot=Key
%    has(potion, healing, _, Key),
%    \+ healthy.
%
%% Azione: Open
%action(dict{cmd:57}) :-
%    near_goal,
%    goal(_, X, Y),
%    openable(X, Y).
%
%% Action eat
%action(dict{cmd:35, slot:Key}) :-  % 64=quaff, slot=Key
%    has(food, _, _, Key),
%    \+ hungry.
%
%% Azione: Open
%action(dict{cmd:57}) :-
%    near_goal,
%    goal(_, X, Y),
%    openable(X, Y).
%
%% Action close
%action(cmd, dict{cmd:cmd}) :-
%    near_goal,
%    escape(_,true).
%
%
%
%
%%walkable((46,7), true).
%%walkable((43,3), false).
%%walkable((45,7), false).
%%walkable((124,7), false).
%
%%openable(43, 3).
%%openable(4, 3).
%
%health(10).
%health_max(10).
%hunger(2).
%
%has(potion, healing, _, b).
%% Stato attuale del agente (posizione)
%agent_pos(2,1).
%
%% Goal (posizione da raggiungere)
%goal(pos(1,1), 4, 3).