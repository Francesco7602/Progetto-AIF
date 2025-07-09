:- dynamic risk_zone_symbol/2.
:- dynamic risk_zone_pos/2.
:- dynamic walkable/2.
:- dynamic health/1.
:- dynamic health_max/1.
:- dynamic has/6.
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

% escape(X,Y) X simbol monster and Y true/false
:- dynamic escape/2.
escape(_,false).

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

action(_, C, dict{cmd:C}).





% beliefSeeMonster(S, HealthBefore, HealthAfter, Danger) is true and returns the monster  danger value
% if symbol S is a known monster;
% if S is unknown and HP decreased, returns true and Danger is the HP loss;
% returns false otherwise.
:- dynamic beliefSeeMonster/4.

:- dynamic maybe_monster/2.

beliefSeeMonster(S, Hp, HpOld, Danger) :-
    (is_monster(S, D); maybe_monster(S, D)),
    T is HpOld - Hp,
    Danger is max(T, D).


beliefSeeMonster(S, Hp, HpOld, Danger) :-
    \+ is_known(_, S, _),
    Danger is HpOld - Hp,
    Danger>0.


:- dynamic use/2.

use(Y, 'q') :-
    has('potion gain level',_,_,_,_,Y).

:- dynamic pos_monster/2.

use(Y, 'q') :-
    has('potions of healing',_,_,_,_,Y),
    \+ healthy,
    pos_monster(A,B),
    agent_pos(X,Y),
    DX is abs(A - X),
    DY is abs(B - Y),
    D is max(DX, DY),
    D >4.

use(Y, 'q') :-
    has('potion of healing',_,_,_,_,Y),
    \+ healthy,
    pos_monster(A,B),
    agent_pos(X,Y),
    DX is abs(A - X),
    DY is abs(B - Y),
    D is max(DX, DY),
    D >4.


use(Y, 'w') :-
    has('samurai sword',_,_,_,'false',Y).

use(Y, 'w') :-
    has('long samurai sword',_,_,_,'false',Y).

%use(Y, 'T') :-
%    has('robe',_,_,_,'true',Y),
%    has('ring mail',_,_,_,'false',_).
%
%use(Y, 'W') :-
%    has('ring mail',_,_,_,'false',Y).



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