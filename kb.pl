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

:- dynamic weapon/2.

:- dynamic use/2.

use(Y, 'q') :-
    has('potion', State, _, _, _, Y),
    State \= 'cursed'.

:- dynamic pos_monster/2.

use(Y, 'q') :-
    has('healing',_,_,_,_,Y),
    \+ healthy.

use(Y, 'w') :-
    has(Nome, _, _, _, false, Y),
    weapon(Nome, Punti),
    \+ (has(Nome2, _, _, _, false, _), weapon(Nome2, Punti2), Punti2 > Punti).
